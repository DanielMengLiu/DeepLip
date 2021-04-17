from yaml import Loader as CLoader
import kaldiio
import sys
import random
import yaml
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
import time
import os
from torch.optim import lr_scheduler
from torchsummary import summary
import logging
import warnings
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import pprint
import utils.utils as utils
from utils.loss import *
import sys
import plda
import joblib
warnings.filterwarnings('ignore')

class Trainer(object):
    def __init__(self):
        f = open('./conf/config.yaml', 'r')
        opts = yaml.load(f, Loader = CLoader)
        f.close()
        self.train_opts = opts['train']
        self.model_opts = opts['model']
        self.data_opts = opts['data']
        self.test_opts = opts['test']

        if self.data_opts['data_format'] == 'kaldi': 
            import data.kaldi_datasets as datasets
            self.trainset = datasets.KaldiTrainDataset(self.data_opts)
            self.voxtestset = datasets.KaldiTestDataset(self.data_opts)
        elif self.data_opts['data_format'] == 'python':
            import data.datasets as datasets
            if self.train_opts['train_type'] == 'None':
                self.trainset = datasets.SpkTrainDataset(self.data_opts)
            elif self.train_opts['train_type'] == 'finetune':
                self.trainset = datasets.TcdtimitTrainDataset(self.data_opts)
            self.voxtestset = datasets.VoxTestset(self.data_opts)
            self.lomgridtestset = datasets.LomgridTestset(self.data_opts)
            self.lomgriddevset = datasets.LomgridDevset(self.data_opts)
            self.gridtestset = datasets.GridTestset(self.data_opts)
        else:
            raise NotImplementedError('Other data formats are not implemented!')

        n_spk = self.trainset.n_spk

        if self.model_opts['arch'] == 'tdnn' or self.model_opts['arch'] == 'etdnn':
            import models.tdnn as tdnn
            model = tdnn.SpeakerEmbNet(self.model_opts)
        elif self.model_opts['arch'] == 'resnet':
            import models.resnet as resnet
            model = resnet.SpeakerEmbNet(self.model_opts)
        else:
            raise NotImplementedError("Other models are not implemented!")

        if self.model_opts['arch'] == 'tdnn' or self.model_opts['arch'] == 'etdnn':
            summary(model.cuda(), (self.model_opts[self.model_opts['arch']]['input_dim'], 100))
        elif self.model_opts['arch'] == 'resnet':
            summary(model.cuda(), (1, self.model_opts[self.model_opts['arch']]['input_dim'], 100))
        else:
            raise NotImplementedError("Other models are not implemented!")

        device_num = 1

        if self.train_opts['device'] == 'gpu':
            device_ids = self.train_opts['gpus_id']
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in self.train_opts['gpus_id']])
            self.device = torch.device('cuda:'+str(device_ids[0]))
            self.model = torch.nn.DataParallel(model.to(self.device), device_ids = device_ids)
        else:
            self.device = torch.device('cpu')

        self.embedding_dim = self.model_opts[self.model_opts['arch']]['embedding_dim']

        if self.train_opts['collate'] == 'length_varied':
            train_collate_fn = self.trainset.collate_fn
        elif self.train_opts['collate'] == 'kaldi':
            train_collate_fn = self.trainset.kaldi_collate_fn
        else:
            raise NotImplementedError("Other collate method are not implemented!")

        self.trainloader = DataLoader(self.trainset, shuffle = True, collate_fn = train_collate_fn, batch_size = self.train_opts['bs'] * device_num, num_workers = 32)
        self.voxtestloader = DataLoader(self.voxtestset, batch_size = 1, shuffle = False, num_workers = 32)
        self.lomgridtestloader = DataLoader(self.lomgridtestset, batch_size = 1, shuffle = False, num_workers = 32)
        self.gridtestloader = DataLoader(self.gridtestset, batch_size = 1, shuffle = False, num_workers = 32)
        self.lomgriddevloader = DataLoader(self.lomgriddevset, batch_size = 1, shuffle = False, num_workers = 32)

        if self.train_opts['loss'] == 'CrossEntropy':
            self.criterion = CrossEntropy(self.embedding_dim, n_spk).to(self.device)
        elif self.train_opts['loss'] == 'LMCL':
            margin_range = self.train_opts['margin']
            self.init_margin = margin_range[0]
            self.end_margin = margin_range[1]
            if self.model_opts['arch'] in ['tdnn', 'etdnn']:
                self.criterion = LMCL(self.embedding_dim, n_spk, self.train_opts['scale'], self.init_margin).to(self.device)
            elif self.model_opts['arch'] == 'resnet':
                self.criterion = LMCL(self.embedding_dim, n_spk, self.train_opts['scale'], self.end_margin).to(self.device)
        else:
            raise NotImplementedError("Other loss function has not been implemented yet!")
         
        param_groups = [{'params': self.model.parameters()}, {'params': self.criterion.parameters()}]

        if self.train_opts['train_type'] != 'finetune':
            if self.train_opts['type'] == 'sgd':
                optim_opts = self.train_opts['sgd']
                self.optim = optim.SGD(param_groups, optim_opts['init_lr'], momentum = optim_opts['momentum'], weight_decay = optim_opts['weight_decay'])
            elif self.train_opts['type'] == 'adam':
                optim_opts = self.train_opts['adam']
                self.optim = optim.Adam(param_groups, optim_opts['init_lr'], weight_decay = optim_opts['weight_decay'])

        self.epoch = self.train_opts['epoch']
        self.resume = self.train_opts['resume']
        self.log_time = time.asctime(time.localtime(time.time())).replace(' ', '_')[4:]
        #  self.lr_scheduler = lr_scheduler.StepLR(self.optim, step_size = self.train_opts['lr_decay_step'], gamma = 0.1)
        if self.train_opts['train_type'] != 'finetune':        
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.optim, milestones = self.train_opts['lr_decay_step'], gamma = 0.1)
        self.current_epoch = 0

        # for kaldi format data read and write
        self.kaldi_helper = utils.KaldiHelper()

        if self.train_opts['train_type'] != 'finetune' and 'resume' in self.train_opts and os.path.exists(self.train_opts['resume']):
            self.load(self.train_opts['resume'])
        elif 'resume' in self.train_opts and os.path.exists(self.train_opts['resume']):
            self.load_finetune(self.train_opts['resume'], param_groups)

    def _adjust_margin(self):
        if self.current_epoch <= 5:
            self.criterion.margin = self.init_margin
        elif self.current_epoch <= self.epoch:
            self.criterion.margin = self.end_margin
    #
    #  def _adjust_lr(self):
    #      if self.current_epoch <= 7:
    #          lr = 0.3
    #      elif self.current_epoch <= 12:
    #          lr = 0.03
    #      elif self.current_epoch <= 16:
    #          lr = 0.003
    #
    #      for param_groups in self.optim.param_groups:
    #          param_groups['lr'] = lr

    def _train(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch + 1, self.epoch + 1):
            self.current_epoch = epoch
            #  self._adjust_lr()
            if self.model_opts['arch'] in ['tdnn', 'etdnn']:
                # lmcl adjust margin
                self._adjust_margin()
            self._train_epoch()
            # self.extract_test_xv()
            # eer, threshold = utils.eer(self.log_time)
            # print("EER: {:.6f}%".format(eer * 100))
            self.lr_scheduler.step()
            #  if self.current_epoch % 5 == 0 or self.current_epoch + 5 > self.epoch:
            self.save()

    def _train_epoch(self):
        self.model.train()
        sum_loss, sum_samples, correct = 0, 0, 0
        progress_bar = tqdm(self.trainloader)
        if self.model_opts['arch'] == 'tdnn' or \
           self.model_opts['arch'] == 'etdnn' or \
           self.model_opts['arch'] == 'resnet':
            for batch_idx, (inputs_feat, targets_label) in enumerate(progress_bar):
                self.optim.zero_grad()
                if self.model_opts['arch'] == 'resnet':
                    inputs_feat = inputs_feat.unsqueeze(1)
                inputs_feat = inputs_feat.to(self.device)
                targets_label = targets_label.to(self.device)
                output = self.model(inputs_feat) # output of 2nd fc layer
                loss, logits = self.criterion(output, targets_label)
                # if torch.isnan(loss):
                #     print(targets_label)
                #     continue
                #     self.load('exp/{}/net.pth'.format(self.log_time))
                #     for param_groups in self.optim.param_groups:
                #         param_groups['lr'] *= 0.1
                #     break
                sum_samples += len(inputs_feat)
                _, prediction = torch.max(logits, dim = 1)
                correct += (prediction == targets_label).sum().item()
                loss.backward()
                self.optim.step()
                sum_loss += loss.item() * len(targets_label)
                progress_bar.set_description(
                        'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] Loss: {:.4f} Acc: {:.4f}%'.format(
                            self.current_epoch, batch_idx + 1, len(self.trainloader),
                            100. * (batch_idx + 1) / len(self.trainloader),
                            sum_loss / sum_samples, 100. * correct / sum_samples
                            )
                        )
            torch.save({'epoch': self.current_epoch, 'state_dict': self.model.state_dict(), 'criterion': self.criterion,
                'optimizer': self.optim.state_dict()},
                'exp/{}/net.pth'.format(self.log_time))

        else:
            raise NotImplementedError("Training process of other models is not implemented!")
        
    def model_average(self, avg_num = 4):
        model_state_dict = {}
        for i in range(avg_num):
            suffix = self.epoch - i
            ckpt = torch.load('exp/{}/net_{}.pth'.format(self.log_time, suffix))
            state_dict = ckpt['state_dict']
            for k, v in state_dict.items():
                if k in model_state_dict:
                    model_state_dict[k] += v
                else:
                    model_state_dict[k] = v
        for k, v in model_state_dict.items():
            model_state_dict[k] = v / avg_num
        torch.save({'epoch': 0, 'state_dict': model_state_dict,
                    'optimizer': ckpt['optimizer']},
                    'exp/{}/net_avg.pth'.format(self.log_time))
        self.model.load_state_dict(model_state_dict)

    def extract_train_xv(self):
        if os.path.exists('exp/{}/net_avg.pth'.format(self.log_time)):
            self.load('exp/{}/net_avg.pth'.format(self.log_time))
        parallel_model = self.model
        self.model = self.model.module
        self.model.eval()
        os.makedirs('exp/{}/train_xv'.format(self.log_time), exist_ok = True)
        with torch.no_grad():
            if self.model_opts['arch'] == 'tdnn' or \
               self.model_opts['arch'] == 'etdnn' or \
               self.model_opts['arch'] == 'resnet':
                for feat, utt in tqdm(self.trainset()):
                    if self.model_opts['arch'] == 'resnet':
                        feat = feat.unsqueeze(1)
                    feat = feat.to(self.device)
                    if self.train_opts['loss'] == 'CrossEntropy':
                        _, xv = self.model.extract_embedding(feat)
                    elif self.train_opts['loss'] == 'LMCL':
                        xv, _ = self.model.extract_embedding(feat)
                    xv = xv.cpu().numpy()
                    train_spk_dir = os.path.join('exp/{}/train_xv'.format(self.log_time), os.path.dirname(utt))
                    os.makedirs(train_spk_dir, exist_ok = True)
                    np.save(os.path.join(train_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)
            else:
                raise NotImplementedError("Extracting process of other models is not implemented!")
        self.model = parallel_model

    def save(self):
        torch.save({'epoch': self.current_epoch, 'state_dict': self.model.state_dict(), 'criterion': self.criterion,
                    'optimizer': self.optim.state_dict()},
                    'exp/{}/net_{}.pth'.format(self.log_time, self.current_epoch))

    def load(self, resume):
        print('loading model from {}'.format(resume))
        self.log_time = resume.split('/')[1]
        ckpt = torch.load(resume)
        self.model.load_state_dict(ckpt['state_dict'])
        # if 'criterion' in ckpt:
        #     self.criterion = ckpt['criterion']
        # self.optim.load_state_dict(ckpt['optimizer'])
        self.current_epoch = ckpt['epoch']
    
    def load_finetune(self, resume, param_groups):
        print('loading model from {}'.format(resume))
        self.log_time = resume.split('/')[1]
        ckpt = torch.load(resume)
        self.model.load_state_dict(ckpt['state_dict'])
        print('model parameter:')
        for param in self.model.parameters():
            param.requires_grad = False
            print(param)
        param_groups = [{'params': self.criterion.parameters()}]
        if self.train_opts['type'] == 'sgd':
            optim_opts = self.train_opts['sgd']
            self.optim = optim.SGD(param_groups, optim_opts['init_lr'], momentum = optim_opts['momentum'], weight_decay = optim_opts['weight_decay'])
        elif self.train_opts['type'] == 'adam':
            optim_opts = self.train_opts['adam']
            self.optim = optim.Adam(param_groups, optim_opts['init_lr'], weight_decay = optim_opts['weight_decay'])
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optim, milestones = self.train_opts['lr_decay_step'], gamma = 0.1)
        print('criterion parameter:')
        for param in self.criterion.parameters():
            print(param.requires_grad)
            print(param)

    def train_plda(self):
        if os.path.exists('exp/{}/net_avg.pth'.format(self.log_time)):
            self.load('exp/{}/net_avg.pth'.format(self.log_time))
        parallel_model = self.model
        self.model = self.model.module
        self.model.eval()
        print('Training PLDA: ')
        # self._print_config(self.lomgriddevset.opts)
        os.makedirs('exp/{}/dev_xv_lomgrid'.format(self.log_time), exist_ok = True)
        with torch.no_grad():
            if self.model_opts['arch'] == 'tdnn' or \
               self.model_opts['arch'] == 'etdnn' or \
               self.model_opts['arch'] == 'resnet':
                for feat, utt in tqdm(self.lomgriddevloader):
                    if self.model_opts['arch'] == 'resnet':
                        feat = feat.unsqueeze(1)
                    feat = feat.to(self.device)
                    utt = utt[0]
                    if self.train_opts['loss'] == 'CrossEntropy':
                        _, xv = self.model.extract_embedding(feat) # cross entropy loss use 1st fc layer output as speaker embedding
                    elif self.train_opts['loss'] == 'LMCL':
                        xv, _ = self.model.extract_embedding(feat) # lmcl use 2nd fc layer output as speaker embedding
                        xv = F.normalize(xv)
                    xv = xv.cpu().numpy()
                    test_spk_dir = 'exp/{}/dev_xv_lomgrid'.format(self.log_time)
                    os.makedirs(test_spk_dir, exist_ok = True)
                    np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)
            else:
                raise NotImplementedError("Extracting process of other models is not implemented!")
        self.model = parallel_model
        filelist = open('./data/manifest/lomgrid_devlist_audio').read().splitlines()
        embeddings = []
        labels = []
        for i in range(0, len(filelist)):
            os.path.join(test_spk_dir, os.path.basename(filelist[i].replace('.wav', '.npy')))
            embedding = np.load(os.path.join(test_spk_dir, os.path.basename(filelist[i].replace('.wav', '.npy'))))
            label = int(filelist[i].split('/')[-1].split('_')[0].replace('s',''))
            embeddings.append(embedding)
            labels.append(label)     
        embeddings = np.array(embeddings).squeeze(1)
        labels = np.array(labels)
        better_classifier = plda.Classifier()
        better_classifier.fit_model(embeddings, labels, n_principal_components=20)
        joblib.dump(better_classifier, 'exp/plda.pkl')

    def extract_test_xv(self):
        if os.path.exists(self.resume) \
           and self.train_opts['train_type'] != 'finetune':
            self.load(self.resume)
        parallel_model = self.model
        self.model = self.model.module
        self.model.eval()
        print('Extracting test embeddings for Voxceleb 1: ')
        # self._print_config(self.voxtestset.opts)
        os.makedirs('exp/{}/test_xv'.format(self.log_time), exist_ok = True)
        with torch.no_grad():
            if self.model_opts['arch'] == 'tdnn' or \
               self.model_opts['arch'] == 'etdnn' or \
               self.model_opts['arch'] == 'resnet':
                for feat, utt in tqdm(self.voxtestloader):
                    if self.model_opts['arch'] == 'resnet':
                        feat = feat.unsqueeze(1)
                    feat = feat.to(self.device)
                    utt = utt[0]
                    if self.train_opts['loss'] == 'CrossEntropy':
                        _, xv = self.model.extract_embedding(feat) # cross entropy loss use 1st fc layer output as speaker embedding
                    elif self.train_opts['loss'] == 'LMCL':
                        xv, _ = self.model.extract_embedding(feat) # lmcl use 2nd fc layer output as speaker embedding
                        xv = F.normalize(xv)
                    xv = xv.cpu().numpy()
                    test_spk_dir = os.path.join('exp/{}/test_xv'.format(self.log_time), os.path.dirname(utt))
                    os.makedirs(test_spk_dir, exist_ok = True)
                    np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)
            else:
                raise NotImplementedError("Extracting process of other models is not implemented!")
        self.model = parallel_model

    def extract_test_xv_lomgrid(self):
        if os.path.exists(self.resume) \
           and self.train_opts['train_type'] != 'finetune':
            self.load(self.resume)
        parallel_model = self.model
        self.model = self.model.module
        self.model.eval()
        print('Extracting test embeddings for LOMBARDGRID: ')
        # self._print_config(self.voxtestset.opts)
        os.makedirs('exp/{}/test_xv_lomgrid'.format(self.log_time), exist_ok = True)
        with torch.no_grad():
            if self.model_opts['arch'] == 'tdnn' or \
               self.model_opts['arch'] == 'etdnn' or \
               self.model_opts['arch'] == 'resnet':
                for feat, utt in tqdm(self.lomgridtestloader):
                    if self.model_opts['arch'] == 'resnet':
                        feat = feat.unsqueeze(1)
                    feat = feat.to(self.device)
                    utt = utt[0]
                    if self.train_opts['loss'] == 'CrossEntropy':
                        _, xv = self.model.extract_embedding(feat) # cross entropy loss use 1st fc layer output as speaker embedding
                    elif self.train_opts['loss'] == 'LMCL':
                        xv, _ = self.model.extract_embedding(feat) # lmcl use 2nd fc layer output as speaker embedding
                        xv = F.normalize(xv)
                    xv = xv.cpu().numpy()
                    test_spk_dir = os.path.join('exp/{}/test_xv_lomgrid'.format(self.log_time), os.path.dirname(utt))
                    os.makedirs(test_spk_dir, exist_ok = True)
                    np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)
            else:
                raise NotImplementedError("Extracting process of other models is not implemented!")
        self.model = parallel_model

    def extract_test_xv_grid(self):
        if os.path.exists(self.resume) \
           and self.train_opts['train_type'] != 'finetune':
            self.load(self.resume)
        parallel_model = self.model
        self.model = self.model.module
        self.model.eval()
        print('Extracting test embeddings for GRID: ')
        # self._print_config(self.voxtestset.opts)
        os.makedirs('exp/{}/test_xv_grid'.format(self.log_time), exist_ok = True)
        with torch.no_grad():
            if self.model_opts['arch'] == 'tdnn' or \
               self.model_opts['arch'] == 'etdnn' or \
               self.model_opts['arch'] == 'resnet':
                for feat, utt in tqdm(self.gridtestloader):
                    if self.model_opts['arch'] == 'resnet':
                        feat = feat.unsqueeze(1)
                    feat = feat.to(self.device)
                    utt = utt[0]
                    if self.train_opts['loss'] == 'CrossEntropy':
                        _, xv = self.model.extract_embedding(feat) # cross entropy loss use 1st fc layer output as speaker embedding
                    elif self.train_opts['loss'] == 'LMCL':
                        xv, _ = self.model.extract_embedding(feat) # lmcl use 2nd fc layer output as speaker embedding
                        xv = F.normalize(xv)
                    xv = xv.cpu().numpy()
                    test_spk_dir = os.path.join('exp/{}/test_xv_grid'.format(self.log_time), os.path.dirname(utt))
                    os.makedirs(test_spk_dir, exist_ok = True)
                    np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)
            else:
                raise NotImplementedError("Extracting process of other models is not implemented!")
        self.model = parallel_model

    def _print_config(self, opts):
        pp = pprint.PrettyPrinter(indent = 2)
        pp.pprint(opts)

    def transform_from_kaldi_xv(self):
        os.makedirs('exp/{}/kaldi_test_xv/'.format(self.log_time), exist_ok = True)
        for xv, utt in self.kaldi_helper('scp:/groups1/gcc50479/spk/code/kaldi_xv/exp/xvector_nnet_1a/xvectors_voxceleb1_test/xvector.scp'):
            utt = utt.split('-')
            utt = '/'.join([utt[0], '-'.join(utt[1:-1]), utt[-1]])
            test_spk_dir = os.path.join('exp/{}/kaldi_test_xv'.format(self.log_time), os.path.dirname(utt))
            os.makedirs(test_spk_dir, exist_ok = True)
            np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), xv)

    def transform_to_kaldi_xv(self, mode = 'test'):
        utt2xv = OrderedDict()
        augment_type = ['reverb', 'music', 'babble', 'noise']
        if mode == 'test':
            f = open('/groups1/gcc50479/spk/code/kaldi_xv/exp/xvector_nnet_1a/xvectors_voxceleb1_test/ori_xvector.scp', 'r')
        else:
            f = open('/groups1/gcc50479/spk/code/kaldi_xv/exp/xvector_nnet_1a/xvectors_train/ori_xvector.scp', 'r')
        for line in tqdm(f):
            line = line.rstrip()
            line = line.split(' ')
            ori_utt, xv_path = line
            utt = ori_utt.split('-')
            if utt[-1] in augment_type:
                utt = '/'.join([utt[0], '-'.join(utt[1:-2]), utt[-1], utt[-2]])
            else:
                utt = '/'.join([utt[0], '-'.join(utt[1:-1]), utt[-1]])
            xv_path = 'exp/{}/{}_xv/{}'.format(self.log_time, mode, utt + '.npy')
            utt2xv[ori_utt] = np.load(xv_path)
        f.close()
        self.kaldi_helper.write_speaker_embedding(utt2xv, 'ark,scp:{}_xvector.ark,{}_xvector.scp'.format(mode, mode))
        
    def __call__(self):
        print("[LOG Time: {}]".format(self.log_time))
        print("Data opts: ")
        self._print_config(self.data_opts)
        print("Model opts: ")
        self._print_config(self.model_opts)
        print("Train opts: ")
        self._print_config(self.train_opts)
        os.makedirs('exp/{}'.format(self.log_time), exist_ok = True)
        self._train()

if __name__ == '__main__':
    mode = 'av_fusion'
    trainer = Trainer()
    if mode == 'train':
        trainer()
        # print('model parameter:')
        # for param in trainer.model.parameters():
        #     print(param.requires_grad)
        #     print(param)
        # print('criterion parameter:')
        # for param in trainer.criterion.parameters():
        #     print(param.requires_grad)
        #     print(param)
        trainer.model_average()
        trainer.extract_test_xv()
        eer, threshold = utils.eer(trainer.log_time)
        print("EER: {:.6f}%".format(eer * 100))
    # voxceleb dataset test
    elif mode == 'test': 
        trainer.extract_test_xv()
        eer, threshold = utils.eer(trainer.log_time)
        print("EER: {:.6f}%".format(eer * 100))
    # audio-visual dataset test
    elif mode == 'av_test':
        if trainer.test_opts['train_plda']:
            trainer.train_plda()
        if trainer.test_opts['eval_lomgrid']:
            trainer.extract_test_xv_lomgrid()
            if trainer.test_opts['use_cos']:              
                eer, threshold = utils.eer_cos_lomgrid(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))
            if trainer.test_opts['use_plda']:              
                eer, threshold = utils.eer_plda_lomgrid(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))
        if trainer.test_opts['eval_grid']:
            trainer.extract_test_xv_grid() 
            if trainer.test_opts['use_cos']:
                eer, threshold = utils.eer_cos_grid(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))
            if trainer.test_opts['use_plda']:
                eer, threshold = utils.eer_plda_grid(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))
    elif mode == 'av_fusion':
        # if trainer.test_opts['train_plda']:
        #     trainer.train_plda()
        if trainer.test_opts['eval_lomgrid']:
            # trainer.extract_test_xv_lomgrid()
            if trainer.test_opts['use_cos']:              
                eer, threshold = utils.eer_cos_lomgrid_featurefusion(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))
            if trainer.test_opts['use_plda']:              
                eer, threshold = utils.eer_plda_lomgrid(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))
        if trainer.test_opts['eval_grid']:
            # trainer.extract_test_xv_grid() 
            if trainer.test_opts['use_cos']:
                eer, threshold = utils.eer_cos_grid_featurefusion(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))
            if trainer.test_opts['use_plda']:
                eer, threshold = utils.eer_plda_grid(trainer.log_time)
                print("EER: {:.6f}%".format(eer * 100))

