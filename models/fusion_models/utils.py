import numpy as np
import kaldiio
from itertools import combinations
import torch
import torch.nn.functional as F
import yaml
from collections import OrderedDict
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
from yaml import Loader as CLoader
import joblib
from glob import glob

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None # choose the hardest negative


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0] # get all index whose the loss > 0
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None # random choose a negative


def semihard_negative(loss_values, margin):
    # get all index whose distance > 0 but < margin
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    # random choose the negative
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        cos_matrix = F.linear(embeddings, embeddings) # get cosine_distance_matrix
        cos_matrix = cos_matrix.cpu()

        labels = labels.cpu().data.numpy() # labels size = (batch_size)
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_cosine = cos_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_cos in zip(anchor_positives, ap_cosine):
                loss_values = cos_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin - ap_cos
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        try:    
            if len(triplets) == 0:
                triplets.append([anchor_positives[0], anchor_positive[1], negative_indices[0]])
        except:
            raise UnboundLocalError

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=hardest_negative,
                                           cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=lambda x: semihard_negative(x, margin),
                                           cpu=cpu)

class Args(object):
    def __init__(self, config_path = 'conf/config.yaml'):
        super(Args, self).__init__()
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader = CLoader)
        self.opts = OrderedDict()
        data_opts = config['data']
        model_opts = config['model']
        train_opts = config['train']
        self.parse_opts(data_opts)
        self.parse_opts(model_opts)
        self.parse_opts(train_opts)

    def parse_opts(self, opts):
        keys = opts.keys()
        values = opts.values()
        for key in keys:
            if key in values:
                for k, v in opts[key].items():
                    self.opts[k] = v
            if not isinstance(opts[key], dict):
                self.opts[key] = opts[key]

    def print(self):
        print('{', end = ' ')
        for idx, (k, v) in enumerate(self.opts.items()):
            if idx == 0:
                print('{:20s}: {}'.format(k, v))
            else:
                print('  {:20s}: {}'.format(k, v))
        print('}')

class KaldiHelper(object):
    def __init__(self):
        super(KaldiHelper, self).__init__()

    def read_feat(self, rspecifier):
        '''
        Read feature matrix from Kaldi data. Note the header of Kaldi file is FM

        params:
            rspecifier: Kaldi rspecifier
        return:
            a generator extract feature from Kaldi data
        '''
        with kaldiio.ReadHelper(rspecifier) as reader:
            for uttid, feature in reader:
                yield feature, uttid

    def write_feat(self, utt2feat, wspecifier):
        '''
        Write features into Kaldi format data.

        params:
            utt2feat: a dict, utt -> feature
        return:
            None
        '''
        with kaldiio.WriteHelper(wspecifier) as writer:
            for uttid, feature in utt2feat.items():
                writer(uttid, feature)

    def read_speaker_embedding(self, rspecifier):
        '''
        Read embedding vector from Kaldi data. Note the header of Kaldi file is FV

        params:
            rspecifier: Kaldi rspecifier
        return:
            a generator extract speaker embedding from Kaldi data
        '''
        with kaldiio.ReadHelper(rspecifier) as reader:
            for uttid, vector in reader:
                vector = vector.reshape(-1)
                yield vector, uttid

    def write_speaker_embedding(self, utt2xv, wspecifier):
        '''
        Write embedding vector into Kaldi format data.

        params:
            utt2xv: a dict, utt -> xvector
        return:
            None
        '''
        with kaldiio.WriteHelper(wspecifier) as writer:
            for uttid, vector in utt2xv.items():
                vector = vector.reshape(-1)
                writer(uttid, vector)

def eer(exp_dir):
    y_true = []
    y_pred = []
    with open('task.txt', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_xv'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_xv'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
            y_pred.append(score)
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_cos_lomgrid(exp_dir):
    y_true = []
    y_pred = []
    with open('data/data_audio/trial_lomgrid_2w.txt', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_em_lomgrid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_em_lomgrid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
            y_pred.append(score)
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_cos_grid(exp_dir):
    y_true = []
    y_pred = []
    with open('data/data_audio/trial_grid_2w.txt', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_em_grid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_em_grid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
            y_pred.append(score)
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_plda_lomgrid(exp_dir):
    better_classifier = joblib.load('exp/plda.pkl')
    y_true = []
    y_pred = []
    with open('data/trial/A_lomgrid_trial_2w', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_xv_lomgrid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_xv_lomgrid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            em = np.array([utt1_feat, utt2_feat])
            em =em.squeeze(1)
            U_model = better_classifier.model.transform(em, from_space='D', to_space='U_model')
            U_datum_0 = U_model[0][None,]
            U_datum_1 = U_model[1][None,]
            score = better_classifier.model.calc_same_diff_log_likelihood_ratio(U_datum_0, U_datum_1)
            y_pred.append(score)
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_plda_grid(exp_dir):
    better_classifier = joblib.load('exp/plda.pkl')
    y_true = []
    y_pred = []
    with open('data/trial/A_grid_trial_2w', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_xv_grid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_xv_grid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            em = np.array([utt1_feat, utt2_feat])
            em =em.squeeze(1)
            U_model = better_classifier.model.transform(em, from_space='D', to_space='U_model')
            U_datum_0 = U_model[0][None,]
            U_datum_1 = U_model[1][None,]
            score = better_classifier.model.calc_same_diff_log_likelihood_ratio(U_datum_0, U_datum_1)
            y_pred.append(score)
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_cos_lomgrid_scorefusion(exp_dir):
    y_true = []
    y_pred = []
    with open('data/trial/A_lomgrid_trial_2w', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_xv_lomgrid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_xv_lomgrid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
            y_pred.append(score)
    y_pred = np.array(y_pred)
    y_pred = 0.5 * y_pred

    data_dir = '/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_lombardgrid/'
    filelist = open('/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/preprocessing/lombardgrid_trial_2w').read().splitlines()
    filelist_1 = []
    filelist_2 = []

    for fl in filelist:
        filelist_1.append(fl.split('\t')[0])
        filelist_2.append(fl.split('\t')[1])

    score_list_all = []

    for i in tqdm(range(0, len(filelist))):
        pattern_1 = (data_dir + filelist_1[i] + '*').replace('datasets','embedding')
        filelist_11 = sorted(glob(pattern_1))
        pattern_2 = (data_dir + filelist_2[i] + '*').replace('datasets','embedding')
        filelist_22 = sorted(glob(pattern_2))

        embedding_1_mean = 0
        for m in filelist_11:
            embedding_1_mean += np.mean(np.load(m)['data'].squeeze(-3),0)
        embedding_1_mean = embedding_1_mean / len(filelist_11)
        embedding_2_mean = 0
        for n in filelist_22:
            embedding_2_mean += np.mean(np.load(n)['data'].squeeze(-3),0)
        embedding_2_mean = embedding_2_mean / len(filelist_22)

        score = F.cosine_similarity(torch.from_numpy(embedding_1_mean),torch.from_numpy(embedding_2_mean),dim=0,eps=1e-08)
        score_list_all.append([score.detach().numpy()])
    
    score_list_all = np.array(score_list_all)
    score_list_all = 0.5 * score_list_all
    y_pred = y_pred + score_list_all
    y_pred = list(y_pred)
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_cos_grid_scorefusion(exp_dir):
    y_true = []
    y_pred = []
    with open('data/trial/A_grid_trial_2w', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_xv_grid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_xv_grid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
            y_pred.append(score)
    y_pred = np.array(y_pred)
    y_pred = 0.5 * y_pred

    data_dir = '/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_grid/'
    filelist = open('/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/preprocessing/grid/grid_trial_2w').read().splitlines()
    filelist_1 = []
    filelist_2 = []

    for fl in filelist:
        filelist_1.append(fl.split('\t')[0])
        filelist_2.append(fl.split('\t')[1])

    score_list_all = []

    for i in range(0, len(filelist)):
        pattern_1 = (data_dir + filelist_1[i] + '*').replace('datasets','embedding')
        filelist_11 = sorted(glob(pattern_1))
        pattern_2 = (data_dir + filelist_2[i] + '*').replace('datasets','embedding')
        filelist_22 = sorted(glob(pattern_2))

        embedding_1 = 0
        for m in filelist_11:
            embedding_1 += np.mean(np.load(m)['data'].squeeze(-3),0)
        embedding_1 = embedding_1 / len(filelist_11)
        embedding_2 = 0
        for n in filelist_22:
            embedding_2 += np.mean(np.load(n)['data'].squeeze(-3),0)
        embedding_2 = embedding_2 / len(filelist_22)
        
        score = F.cosine_similarity(torch.from_numpy(embedding_1),torch.from_numpy(embedding_2),dim=0,eps=1e-08)
        score_list_all.append([score])
    
    score_list_all = np.array(score_list_all)
    score_list_all = 0.5 * score_list_all
    y_pred = y_pred + score_list_all
    y_pred = list(y_pred)
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_cos_lomgrid_featurefusion(exp_dir):
    data_dir = '/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_lombardgrid/'
    y_true = []
    y_pred = []
    with open('data/trial/A_lomgrid_trial_2w', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_xv_lomgrid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_xv_lomgrid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            
            filelist_1 = test_utt1.split('_')[0] + '/' + test_utt1.replace('.wav', '')
            filelist_2 = test_utt2.split('_')[0] + '/' + test_utt2.replace('.wav', '')
            pattern_1 = (data_dir + filelist_1 + '*').replace('datasets','embedding')
            filelist_11 = sorted(glob(pattern_1))
            pattern_2 = (data_dir + filelist_2 + '*').replace('datasets','embedding')
            filelist_22 = sorted(glob(pattern_2))

            embedding_1_mean = 0
            for m in filelist_11:
                embedding_1_mean += np.mean(np.load(m)['data'].squeeze(-3),0)
            embedding_1_mean = embedding_1_mean / len(filelist_11) 
            embedding_2_mean = 0
            for n in filelist_22:
                embedding_2_mean += np.mean(np.load(n)['data'].squeeze(-3),0)
            embedding_2_mean = embedding_2_mean / len(filelist_22)

            utt1_feat_a = feature_normalize(utt1_feat.reshape(-1))
            utt2_feat_a = feature_normalize(utt2_feat.reshape(-1))
            utt1_feat_v = feature_normalize(embedding_1_mean.reshape(-1))
            utt2_feat_v = feature_normalize(embedding_2_mean.reshape(-1))
            
            utt1_feat = np.hstack((utt1_feat_v, utt1_feat_a))
            utt2_feat = np.hstack((utt2_feat_v, utt2_feat_a))

            score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
            y_pred.append(score)
    f.close()        
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def eer_cos_grid_featurefusion(exp_dir):
    data_dir = '/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_grid/'
    y_true = []
    y_pred = []
    with open('data/trial/A_grid_trial_2w', 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            true_score, test_utt1, test_utt2 = line.split(' ')
            y_true.append(eval(true_score))
            utt1_feat = np.load(os.path.join('exp/{}/test_xv_grid'.format(exp_dir), test_utt1.replace('.wav', '.npy')))
            utt2_feat = np.load(os.path.join('exp/{}/test_xv_grid'.format(exp_dir), test_utt2.replace('.wav', '.npy')))
            
            filelist_1 = test_utt1.replace('.wav', '')
            filelist_2 = test_utt2.replace('.wav', '')
            pattern_1 = (data_dir + filelist_1 + '*').replace('datasets','embedding')
            filelist_11 = sorted(glob(pattern_1))
            pattern_2 = (data_dir + filelist_2 + '*').replace('datasets','embedding')
            filelist_22 = sorted(glob(pattern_2))

            embedding_1_mean = 0
            for m in filelist_11:
                embedding_1_mean += np.mean(np.load(m)['data'].squeeze(-3),0)
            embedding_1_mean = embedding_1_mean / len(filelist_11)
            embedding_2_mean = 0
            for n in filelist_22:
                embedding_2_mean += np.mean(np.load(n)['data'].squeeze(-3),0)
            embedding_2_mean = embedding_2_mean / len(filelist_22)

            utt1_feat_a = feature_normalize(utt1_feat.reshape(-1))
            utt2_feat_a = feature_normalize(utt2_feat.reshape(-1))
            utt1_feat_v = feature_normalize(embedding_1_mean.reshape(-1))
            utt2_feat_v = feature_normalize(embedding_2_mean.reshape(-1))
            
            utt1_feat = np.hstack((utt1_feat_v, utt1_feat_a))
            utt2_feat = np.hstack((utt2_feat_v, utt2_feat_a))
            score = cosine_similarity(utt1_feat.reshape(1, -1), utt2_feat.reshape(1, -1)).reshape(-1)
            y_pred.append(score)
    f.close()        
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold

def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std

if __name__ == '__main__':
    args = Args()
    args.print()
