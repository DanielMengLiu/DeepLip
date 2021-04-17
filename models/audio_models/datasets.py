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

class SpkTrainDataset(Dataset):
    def __init__(self, opts):
        frame_range = opts['frames'] # frame number range in training
        self.lower_frame_num = frame_range[0]
        self.higher_frame_num = frame_range[1]
        TRAIN_MANIFEST = opts['train_manifest']
        opts = opts['audio_config']
        self.rate = opts['rate']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type] # can choose mfcc or fbank as input feat
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
        # change self.count config
        mean_duration_per_utt = (np.mean(frame_range) - 1) * self.opts['win_shift'] + self.opts['win_len']
        self.count = math.floor(total_duration / mean_duration_per_utt)

    def _load_audio(self, path, start, stop):
        y = None
        y, sr = sf.read(path, start = start, stop = stop, dtype = 'float32', always_2d = True)
        y = y[:, 0]
        return y, sr

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

    # TODO multi processes collate function
    def mp_collate_fn(self, batch):
        frame = random.randint(self.lower_frame_num, self.higher_frame_num) # random select a frame number in uniform distribution
        duration = (frame - 1) * self.opts['win_shift'] + self.opts['win_len'] # duration in time of one training speech segment
        samples_num = int(duration * self.rate) # duration in sample point of one training speech segment
        feats = []
        for sid in batch:
            speaker = self.dataset[sid]
            y = []
            n_samples = 0
            while n_samples < samples_num:
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
            feats.append(feat)
        feats = np.array(feats).astype(np.float32)
        labels = np.array(batch).astype(np.int64)
        return torch.from_numpy(feats).transpose_(1, 2), torch.from_numpy(labels)

    def collate_fn(self, batch):
        frame = random.randint(self.lower_frame_num, self.higher_frame_num) # random select a frame number in uniform distribution
        duration = (frame - 1) * self.opts['win_shift'] + self.opts['win_len'] # duration in time of one training speech segment
        samples_num = int(duration * self.rate) # duration in sample point of one training speech segment
        feats = []
        for sid in batch:
            speaker = self.dataset[sid]
            y = []
            n_samples = 0
            while n_samples < samples_num:
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
            feats.append(feat)
        feats = np.array(feats).astype(np.float32)
        labels = np.array(batch).astype(np.int64)
        return torch.from_numpy(feats).transpose_(1, 2), torch.from_numpy(labels)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.n_spk
        return idx

    def __call__(self):
        idx = 0
        wavlist = []
        for i in self.dataset:
            wavlist.extend(i)
        while idx < len(self):
            wav_path, duration, rate = wavlist[idx]
            data, _ = sf.read(wav_path)
            feat = self._extract_feature(data)
            feat = feat.astype(np.float32)
            feat = torch.from_numpy(feat).unsqueeze_(0).transpose_(1,2)
            yield feat, wav_path
            idx += 1

class TcdtimitTrainDataset(Dataset):
    def __init__(self, opts):
        frame_range = opts['frames'] # frame number range in training
        self.lower_frame_num = frame_range[0]
        self.higher_frame_num = frame_range[1]
        TRAIN_MANIFEST = opts['finetune_manifest']
        opts = opts['python_data_config']
        self.rate = opts['rate']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type] # can choose mfcc or fbank as input feat
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
        # change self.count config
        mean_duration_per_utt = (np.mean(frame_range) - 1) * self.opts['win_shift'] + self.opts['win_len']
        self.count = math.floor(total_duration / mean_duration_per_utt)

    def _load_audio(self, path, start, stop):
        y = None
        y, sr = sf.read(path, start = start, stop = stop, dtype = 'float32', always_2d = True)
        y = y[:, 0]
        return y, sr

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

    def collate_fn(self, batch):
        frame = random.randint(self.lower_frame_num, self.higher_frame_num) # random select a frame number in uniform distribution
        duration = (frame - 1) * self.opts['win_shift'] + self.opts['win_len'] # duration in time of one training speech segment
        samples_num = int(duration * self.rate) # duration in sample point of one training speech segment
        feats = []
        for sid in batch:
            speaker = self.dataset[sid]
            y = []
            n_samples = 0
            while n_samples < samples_num:
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
            feats.append(feat)
        feats = np.array(feats).astype(np.float32)
        labels = np.array(batch).astype(np.int64)
        return torch.from_numpy(feats).transpose_(1, 2), torch.from_numpy(labels)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        idx = idx % self.n_spk
        return idx

    def __call__(self):
        idx = 0
        wavlist = []
        for i in self.dataset:
            wavlist.extend(i)
        while idx < len(self):
            wav_path, duration, rate = wavlist[idx]
            data, _ = sf.read(wav_path)
            feat = self._extract_feature(data)
            feat = feat.astype(np.float32)
            feat = torch.from_numpy(feat).unsqueeze_(0).transpose_(1,2)
            yield feat, wav_path
            idx += 1
        
class VoxTestset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz

        '''
        self.root_path = opts['test_root']
        path = opts['test_manifest']
        opts = opts['python_data_config']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type]
        self.utts = []
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utts.append(line[1])
                self.utts.append(line[2])
        self.utts = list(set(self.utts))

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
        return torch.from_numpy(feat).transpose_(0, 1), utt   

class LomgridTestset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz

        '''
        self.root_path = "/data/datasets/Lombard GRID/lombardgrid/audio"
        path = opts['test_manifest_lomgrid']
        opts = opts['python_data_config']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type]
        self.utts = []
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                self.utts.append(line[1])
                self.utts.append(line[2])
        self.utts = list(set(self.utts))

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
        return torch.from_numpy(feat).transpose_(0, 1), utt

class GridTestset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz

        '''
        self.root_path = "/data/datasets/GRID/audio"
        path = opts['test_manifest_grid']
        opts = opts['python_data_config']
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
        return torch.from_numpy(feat).transpose_(0, 1), utt

# plda training
class LomgridDevset(Dataset):
    def __init__(self, opts):
        '''
        default sample rate is 16kHz

        '''
        path = "/data/liumeng/ASV-SOTA/data/manifest/lomgrid_devlist_audio"
        opts = opts['python_data_config']
        self.feat_type = opts['feat_type']
        self.opts = opts[self.feat_type]
        self.utts = []
        with open(path, 'r') as f:
            for line in f:
                self.utts.append(line)

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
        utt = self.utts[idx].strip()
        data, rate = sf.read(utt)
        feat = self._extract_feature(data, rate)
        return torch.from_numpy(feat).transpose_(0, 1), utt

if __name__ == '__main__':
    import yaml
    from torch.utils.data import DataLoader
    import pprint
    import time
    f = open('./conf/config.yaml', 'r')
    opts = yaml.load(f, Loader=yaml.CLoader)['data']
    f.close()
    trainset = SpkTrainDataset(opts)
    print(trainset.n_spk)
    #  for feat, utt_path in trainset():
        #  #  print(feat)
        #  print(utt_path)
    #      break
    #  collate_fn = trainset.collate_fn
    #  train_loader = DataLoader(trainset, batch_size = 128, shuffle = True, collate_fn = collate_fn, num_workers = 32)
    #  start = time.time()
    #  for feat, label in train_loader:
    #      continue
    #  end = time.time()
    #  print('time: {:.4f}'.format(end - start))
    #  trainiter = iter(train_loader)
    #  feats, labels = next(trainiter)
    #  feats = feats.unsqueeze(1)
    #  print(feats.size())
    #  pp = pprint.PrettyPrinter(indent = 2)
    #  pp.pprint(trainset.opts)

    #  voxtestset = VoxTestset(opts)
    #  test_loader = DataLoader(voxtestset, batch_size = 1, shuffle = False, num_workers = 4)
    #  for feat, utt in tqdm(test_loader):
    #      continue
    #  testiter = iter(test_loader)
    #  feat, utt = next(testiter)
    #  print(utt[0])
