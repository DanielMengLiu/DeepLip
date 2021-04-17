from random import sample
from tqdm import tqdm
import multiprocessing
import soundfile as sf
import os
from python_speech_features import mfcc, fbank, logfbank, delta
from librosa import stft, magphase
from torch.utils.data import Dataset
import torch
import pickle
import math
import random
import numpy as np
import csv
from collections import OrderedDict
import librosa
import glob
from models.video_models.dataloaders import get_preprocessing_pipelines

class SpkTrainDataset(Dataset):
    def __init__(self, opts):
        frame_range = opts['audio_config']['frames'] # frame number range in training
        self.lower_frame_num = frame_range[0]
        self.higher_frame_num = frame_range[1]
        TRAIN_MANIFEST = opts['train_manifest']
        opts = opts['audio_config']
        self.rate = opts['feature_config']['rate']
        self.feat_type = opts['feature_config']['feat_type']
        self.opts = opts['feature_config'][self.feat_type] # can choose mfcc or fbank as input feat
        self.dataset = []
        self.count = 0
        current_sid = -1
        total_duration = 0
        with open(TRAIN_MANIFEST, 'r') as f:
            reader = csv.reader(f)
            for sid, aid, filename, duration, samplerate in reader:
                if sid != current_sid:
                    self.dataset.append([])
                    current_sid = sid
                self.dataset[-1].append((filename, float(duration), int(samplerate)))
                self.count += 1
                total_duration += eval(duration)
        self.n_spk = len(self.dataset)
        # better sample strategy
        mean_duration_per_utt = (np.mean(frame_range) - 1) * self.opts['win_shift'] + self.opts['win_len']
        self.count = math.floor(total_duration / mean_duration_per_utt)

    def _load_audio(self, path, start, stop):
        y = None
        y, sr = sf.read(path, start = start, stop = stop, dtype = 'float32', always_2d = True)
        y = y[:, 0]
        return y, sr

    def _load_video(self, filename):
        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            else:
                return np.load(filename)
        except IOError:
            print( "Error when reading file: {}".format(filename) )
            sys.exit()

    def _normalize(self, feat):
        return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

    def _delta(self, feat, order = 2):
        if order == 2:
            feat_d1 = delta(feat, N = 1)
            feat_d2 = delta(feat, N = 2)
            feat = np.hstack([feat, feat_d1, feat_d2])
        elif order == 1:
            feat_d1 = delta(feat, N = 1)
            feat = np.hstack([feat, feat_d1])
        return feat

    def _extract_feature(self, data):
        if self.feat_type == 'mfcc':
            feat = mfcc(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], numcep = self.opts['num_cep'])
        elif self.feat_type == 'fbank':
            feat, _ = fbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'logfbank':
            feat = logfbank(data, self.rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'stft':
            feat = stft(data, n_fft = self.opts['n_fft'], hop_length = int(self.rate * self.opts['win_shift']), win_length = int(self.rate * self.opts['win_len']))
            feat, _ = magphase(feat)
            feat = np.log1p(feat)
            feat = feat.transpose(1,0)
        else:
            raise NotImplementedError("Other features are not implemented!")
        if self.opts['normalize']:
            feat = self._normalize(feat)
        if self.opts['delta']:
            feat = self._delta(feat, order = 2)
        return feat

    def _get_preprocessing_video(self):
        # -- preprocess for the video stream
        preprocessing = {}
        # -- LRW config
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])

        preprocessing['test'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])
        return preprocessing

    def collate_fn(self, batch):
        # generate audio and video training samples
        feats_audio = []
        feats_video = []
        frame = random.randint(self.lower_frame_num, self.higher_frame_num) # random select a frame number in uniform distribution
        duration = (frame - 1) * self.opts['win_shift'] + self.opts['win_len'] # duration in time of one training speech segment
        samples_num = int(duration * self.rate) # duration in sample point of one training speech segment
        preprocessing_func = get_preprocessing_pipelines()['test']
        for sid in batch:
            speaker = self.dataset[sid]
            y = []
            n_samples = 0
            aid = -1
            while n_samples < samples_num:
                if n_samples == 0:
                    aid = random.randrange(0, len(speaker))
                audio = speaker[aid]
                t, sr = audio[1], audio[2]
                samples_len = int(t * sr)
                start = int(random.uniform(0, t) * sr) # random select start point of speech
                _y, _ = self._load_audio(audio[0], start = start, stop = samples_len) # read speech data from start point to the end
                if _y is not None:
                    y.append(_y)
                    n_samples += len(_y)
            y = np.hstack(y)[:samples_num]
            feat = self._extract_feature(y)
            feats_audio.append(feat)

            video_search = os.path.join('/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_TCDTIMIT',audio[0].split('/')[-3],audio[0].split('/')[-1]).replace('.wav','')
            video_path = sorted(glob.glob(video_search + '*.npz'))
            video_group = []
            for x in video_path:
                a = preprocessing_func(self._load_video(x))
                video_group.append(preprocessing_func(self._load_video(x)))
            video_group = np.array(video_group).astype(np.float32)
            feats_video.append(video_group)
        
        feats_audio = np.array(feats_audio).astype(np.float32)
        #feats_video = np.array(feats_video).astype(np.float32)
        labels = np.array(batch).astype(np.int64)

        return feats_video, torch.from_numpy(feats_audio).transpose_(1, 2), torch.from_numpy(labels)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # speaker balance
        idx = idx % self.n_spk
        return idx

        
class LomgridTestset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz

        '''
        self.root_path = "/data/datasets/Lombard GRID/lombardgrid/audio"
        path = opts['test_trial_lomgrid']
        opts = opts['audio_config']['feature_config']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type]
        self.utts = []
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utts.append(line[1])
                self.utts.append(line[2])
        self.utts = list(set(self.utts))

    def _load_video(self, filename):
        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            else:
                return np.load(filename)
        except IOError:
            print( "Error when reading file: {}".format(filename) )
            sys.exit()

    def _get_preprocessing_video(self):
        # -- preprocess for the video stream
        preprocessing = {}
        # -- LRW config
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])

        preprocessing['test'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])
        return preprocessing

    def _normalize(self, feat):
        return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

    def _delta(self, feat, order = 2):
        if order == 2:
            feat_d1 = delta(feat, N = 1)
            feat_d2 = delta(feat, N = 2)
            feat = np.hstack([feat, feat_d1, feat_d2])
        elif order == 1:
            feat_d1 = delta(feat, N = 1)
            feat = np.hstack([feat, feat_d1])
        return feat

    def _extract_feature(self, data, rate):
        if self.feat_type == 'mfcc':
            feat = mfcc(data, rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], numcep = self.opts['num_cep'])
        elif self.feat_type == 'fbank':
            feat, _ = fbank(data, rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'logfbank':
            feat = logfbank(data, rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'stft':
            feat = stft(data, n_fft = self.opts['n_fft'], hop_length = int(rate * self.opts['win_shift']), win_length = int(rate * self.opts['win_len']))
            feat, _ = magphase(feat)
            feat = np.log1p(feat)
            feat = feat.transpose(1,0)
        else:
            raise NotImplementedError("Other features are not implemented!")

        if self.opts['normalize']:
            feat = self._normalize(feat)
        if self.opts['delta']:
            feat = self._delta(feat, order = 2)
        return feat.astype(np.float32)

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]
        utt_path = os.path.join(self.root_path, utt)
        data, rate = sf.read(utt_path)
        feat = self._extract_feature(data, rate)

        spk = utt.split('_')[0]
        fn = utt.split('.')[0]
        feats_video = []
        preprocessing_func = get_preprocessing_pipelines()['test']
        video_search = os.path.join('/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_lombardgrid', spk, fn)
        video_path = sorted(glob.glob(video_search + '*.npz'))
        video_group = []
        for x in video_path:
            video_group.append(preprocessing_func(self._load_video(x)))
        video_group = np.array(video_group).astype(np.float32)
        feats_video.append(video_group)
        return feats_video, torch.from_numpy(feat).transpose_(0, 1), utt

class GridTestset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz

        '''
        self.root_path = "/data/datasets/GRID/audio"
        path = opts['test_trial_grid']
        opts = opts['audio_config']['feature_config']
        self.rate = opts['rate']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type]
        self.utts = []

        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utts.append(line[1])
                self.utts.append(line[2])
        self.utts = list(set(self.utts))

    def _load_video(self, filename):
        try:
            if filename.endswith('npz'):
                return np.load(filename)['data']
            else:
                return np.load(filename)
        except IOError:
            print( "Error when reading file: {}".format(filename) )
            sys.exit()

    def _get_preprocessing_video(self):
        # -- preprocess for the video stream
        preprocessing = {}
        # -- LRW config
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])

        preprocessing['test'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])
        return preprocessing

    def _normalize(self, feat):
        return (feat - feat.mean(axis = 0)) / (feat.std(axis = 0) + 2e-12)

    def _delta(self, feat, order = 2):
        if order == 2:
            feat_d1 = delta(feat, N = 1)
            feat_d2 = delta(feat, N = 2)
            feat = np.hstack([feat, feat_d1, feat_d2])
        elif order == 1:
            feat_d1 = delta(feat, N = 1)
            feat = np.hstack([feat, feat_d1])
        return feat

    def _extract_feature(self, data, rate):
        if self.feat_type == 'mfcc':
            feat = mfcc(data, rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], numcep = self.opts['num_cep'])
        elif self.feat_type == 'fbank':
            feat, _ = fbank(data, rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'logfbank':
            feat = logfbank(data, rate, winlen = self.opts['win_len'], winstep = self.opts['win_shift'], nfilt = self.opts['num_bin'])
        elif self.feat_type == 'stft':
            feat = stft(data, n_fft = self.opts['n_fft'], hop_length = int(rate * self.opts['win_shift']), win_length = int(rate * self.opts['win_len']))
            feat, _ = magphase(feat)
            feat = np.log1p(feat)
            feat = feat.transpose(1,0)
        else:
            raise NotImplementedError("Other features are not implemented!")

        if self.opts['normalize']:
            feat = self._normalize(feat)
        if self.opts['delta']:
            feat = self._delta(feat, order = 2)
        return feat.astype(np.float32)

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = self.utts[idx]
        utt_path = os.path.join(self.root_path, utt)
        data, rate = sf.read(utt_path)
        data = data.transpose(1,0)[0]
        if rate != self.rate:
            data = librosa.resample(data, rate, self.rate)
        feat = self._extract_feature(data, self.rate)

        spk = utt.split('/')[0]
        fn = utt.split('/')[1].split('.')[0]
        feats_video = []
        preprocessing_func = get_preprocessing_pipelines()['test']
        video_search = os.path.join('/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_grid', spk, fn)
        video_path = sorted(glob.glob(video_search + '*.npz'))
        video_group = []
        for x in video_path:
            video_group.append(preprocessing_func(self._load_video(x)))
        video_group = np.array(video_group).astype(np.float32)
        feats_video.append(video_group)
        return feats_video, torch.from_numpy(feat).transpose_(0, 1), utt