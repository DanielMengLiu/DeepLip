import sys
import os
import time
import random
import yaml
from yaml import Loader as CLoader
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchsummary import summary
import logging
import pprint
import warnings
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import models.audio_models.utils as utils_audio
from models.audio_models.loss import *
import models.fusion_models.datasets as datasets
import models.audio_models.tdnn as tdnn
from models.video_models.model import Lipreading
import models.fusion_models.model_fusion as model_fusion
import models.fusion_models.utils as utils
import models.fusion_models.LBP as LBP
# import models.fusion_models.compact_bilinear_pooling as compact_bilinear_pooling
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling
from glob import glob
warnings.filterwarnings('ignore')

class Trainer(object):
    def __init__(self, mode):
        f = open('./conf/fusion_config.yaml', 'r')
        opts = yaml.load(f, Loader = CLoader)
        f.close()
        self.train_opts = opts['train']
        self.model_opts = opts['model']
        self.data_opts = opts['data']
        self.test_opts = opts['test']
        self.mode = mode

        self.trainset = datasets.SpkTrainDataset(self.data_opts)
        self.lomgridtestset = datasets.LomgridTestset(self.data_opts)
        self.gridtestset = datasets.GridTestset(self.data_opts)

        n_spk = self.trainset.n_spk

        # audio model
        if self.model_opts['audio_config']['arch'] == 'tdnn' or self.model_opts['audio_config']['arch'] == 'etdnn':
            self.model_audio = tdnn.SpeakerEmbNet(self.model_opts['audio_config'])
            #summary(model_audio.cuda(), (self.model_opts['audio_config'][self.model_opts['audio_config']['arch']]['input_dim'], 100))
        else:
            raise NotImplementedError("Other models are not implemented!")
        # video model
        if self.model_opts['video_config']['arch'] == 'tcn':
            extract_feats= self.model_opts['video_config']['tcn']['extract_feats']
            backbone_type = self.model_opts['video_config']['tcn']['backbone_type']
            width_mult = self.model_opts['video_config']['tcn']['width_mult']
            relu_type = self.model_opts['video_config']['tcn']['relu_type']
            tcn_options = { 'num_layers': self.model_opts['video_config']['tcn']['tcn_num_layers'],
                            'kernel_size': self.model_opts['video_config']['tcn']['tcn_kernel_size'],
                            'dropout': self.model_opts['video_config']['tcn']['tcn_dropout'],
                            'dwpw': self.model_opts['video_config']['tcn']['tcn_dwpw'],
                            'width_mult': self.model_opts['video_config']['tcn']['tcn_width_mult'],
                        }
            self.model_video = Lipreading( num_classes=n_spk,
                            tcn_options=tcn_options,
                            backbone_type=backbone_type,
                            relu_type=relu_type,
                            width_mult=width_mult,
                            extract_feats=extract_feats)
            #summary(model_video.cuda(), (self.model_opts['video_config'][self.model_opts['video_config']['arch']]['input_dim'], 100))
        else:
            raise NotImplementedError("Other models are not implemented!")
        # fusion model
        self.embedding_dim = self.model_opts['audio_config'][self.model_opts['audio_config']['arch']]['embedding_dim']
        #self.model_fusion = model_fusion.model_fusion(self.embedding_dim * 2, 512, n_spk, extract_feats=True)
        #self.model_fusion = CompactBilinearPooling(self.embedding_dim, self.embedding_dim, 512)
        self.model_fusion = LBP.LowFER(self.embedding_dim, self.embedding_dim, 512)

        device_num = 1
        if self.train_opts['device'] == 'gpu':
            device_ids = self.train_opts['gpus_id']
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in self.train_opts['gpus_id']])
            self.device = torch.device('cuda:'+str(device_ids[0]))
            self.model_audio = torch.nn.DataParallel(self.model_audio.to(self.device), device_ids = device_ids)
            self.model_video = torch.nn.DataParallel(self.model_video.to(self.device), device_ids = device_ids)
            self.model_fusion = torch.nn.DataParallel(self.model_fusion.to(self.device), device_ids = device_ids)
        else:
            self.device = torch.device('cpu')

        if self.train_opts['audio_config']['collate'] == 'length_varied':
            train_collate_fn = self.trainset.collate_fn
        elif self.train_opts['audio_config']['collate'] == 'None':
            train_collate_fn = None
        else:
            raise NotImplementedError("Other collate functions are not implemented!")

        self.trainloader = DataLoader(self.trainset, shuffle = True, collate_fn = train_collate_fn, \
            batch_size = self.train_opts['bs'] * device_num, num_workers = 32)
        self.lomgridtestloader = DataLoader(self.lomgridtestset, batch_size = 1, shuffle = False, num_workers = 32)
        self.gridtestloader = DataLoader(self.gridtestset, batch_size = 1, shuffle = False, num_workers = 32)

        if self.train_opts['loss'] == 'CrossEntropy':
            self.criterion = CrossEntropy(1536, n_spk).to(self.device)
        elif self.train_opts['loss'] == 'LMCL':
            margin_range = self.train_opts['audio_config']['margin']
            self.init_margin = margin_range[0]
            self.end_margin = margin_range[1]
            if self.model_opts['audio_config']['arch'] in ['tdnn', 'etdnn']:
                self.criterion = LMCL(self.embedding_dim, n_spk, self.train_opts['audio_config']['scale'], self.init_margin).to(self.device)
        else:
            raise NotImplementedError("Other loss function has not been implemented yet!")
         
        param_groups = [{'params': self.model_fusion.parameters()}, {'params': self.criterion.parameters()}]

        if self.train_opts['optimizer'] == 'sgd':
            optim_opts = self.train_opts['sgd']
            self.optim = optim.SGD(param_groups, optim_opts['init_lr'], momentum = optim_opts['momentum'], weight_decay = optim_opts['weight_decay'])

        self.epoch = self.train_opts['epoch']
        self.resume_audio = self.train_opts['audio_config']['resume']
        self.resume_video = self.train_opts['video_config']['resume']
        self.resume_fusion = self.train_opts['resume']
        self.log_time = time.asctime(time.localtime(time.time())).replace(' ', '_')[4:]
        #  self.lr_scheduler = lr_scheduler.StepLR(self.optim, step_size = self.train_opts['lr_decay_step'], gamma = 0.1    
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optim, milestones = self.train_opts['lr_decay_step'], gamma = 0.1)
        self.current_epoch = 0

        if os.path.exists(self.train_opts['audio_config']['resume']) and os.path.exists(self.train_opts['video_config']['resume']):
            self.load_finetune()

    def _adjust_margin(self):
        if self.current_epoch <= 5:
            self.criterion.margin = self.init_margin
        elif self.current_epoch <= self.epoch:
            self.criterion.margin = self.end_margin

    def _train(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch + 1, self.epoch + 1):
            self.current_epoch = epoch
            #  self._adjust_lr()
            # if self.model_opts['audio_config']['arch'] in ['tdnn', 'etdnn']:
            #     # lmcl adjust margin
            #     self._adjust_margin()
            self._train_epoch()
            # self.extract_test_xv()
            # eer, threshold = utils.eer(self.log_time)
            # print("EER: {:.6f}%".format(eer * 100))
            self.lr_scheduler.step()
            #  if self.current_epoch % 5 == 0 or self.current_epoch + 5 > self.epoch:
            self.save()

    def _train_epoch(self):
        parallel_model_fusion = self.model_fusion
        self.model_fusion = self.model_fusion.module
        self.model_fusion.train()

        parallel_model_audio = self.model_audio
        self.model_audio = self.model_audio.module
        self.model_audio.eval()

        parallel_model_video = self.model_video
        self.model_video = self.model_video.module
        self.model_video.eval()

        sum_loss, sum_samples, correct = 0, 0, 0
        progress_bar = tqdm(self.trainloader)
        if (self.model_opts['audio_config']['arch'] == 'tdnn' or \
           self.model_opts['audio_config']['arch'] == 'etdnn') \
           and self.model_opts['video_config']['arch'] == 'tcn':
            for batch_idx, (input_video, input_audio, targets_label) in enumerate(progress_bar):
                self.optim.zero_grad()
                # audio 
                xv_audio, _ = self.model_audio.extract_embedding(input_audio.to(self.device)) # lmcl use 2nd fc layer output as speaker embedding

                # video
                em_video = []
                em_video_mask = []
                for i in range(0,targets_label.shape[0]):
                #for video_group in input_video:
                    em = 0
                    video_group = input_video[i]
                    if len(video_group) > 0: 
                        for v in video_group:
                            v = torch.FloatTensor(torch.from_numpy(v))[None, None, :, :, :].to(self.device)
                            em += torch.mean(self.model_video(v, lengths=[v.shape[0]]).squeeze(-3),dim=0)
                        em_video.append(em / len(video_group))
                        em_video_mask.append(True)
                    else:
                        # operation for bad video-audio pairs
                        em = torch.randn(512).to(self.device)
                        em_video.append(em)
                        em_video_mask.append(False)

                em_video = torch.stack(em_video)

                # remove bad video-audio pairs
                em_video = em_video[em_video_mask]
                xv_audio = xv_audio[em_video_mask]
                targets_label = targets_label[em_video_mask]

                # # # features normalization
                # xv_audio = self.feature_normalize(xv_audio)
                # em_video = self.feature_normalize(em_video)

                # feature fusion 
                output = self.model_fusion(xv_audio, em_video)

                # # features normalization
                # xv_audio = self.feature_normalize(xv_audio)
                # em_video = self.feature_normalize(em_video)

                # output = torch.cat([xv_audio,em_video,output],dim=1)
                # output = (xv_audio + em_video + output) / 3
                output = self.feature_normalize(output).squeeze(0)

                # output = em_video 
                loss, logits = self.criterion(output, targets_label.to(self.device))

                sum_samples += len(input_audio)
                _, prediction = torch.max(logits, dim = 1)
                correct += (prediction == targets_label.to(self.device)).sum().item()
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
            torch.save({'epoch': self.current_epoch, 'state_dict': self.model_fusion.state_dict(), 'criterion': self.criterion,
                'optimizer': self.optim.state_dict()},
                'exp/{}/net.pth'.format(self.log_time))
        else:
            raise NotImplementedError("Training process of other models is not implemented!")
        self.model_audio = parallel_model_audio
        self.model_video = parallel_model_video
        self.model_fusion = parallel_model_fusion
        
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
        self.model_fusion.load_state_dict(model_state_dict)

    def save(self):
        torch.save({'epoch': self.current_epoch, 'state_dict': self.model_fusion.state_dict(), 'criterion': self.criterion,
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
    
    def load_finetune(self):
        print('loading model from {} and {}'.format(self.resume_audio, self.resume_video))
        ckpt_audio = torch.load(self.resume_audio)
        ckpt_video = torch.load(self.resume_video)
        self.model_audio.load_state_dict(ckpt_audio['state_dict'])
        self.model_video.load_state_dict(ckpt_video)

        if self.mode == 'av_test':
            ckpt_fusion = torch.load(self.resume_fusion)
            self.model_fusion.load_state_dict(ckpt_fusion['state_dict'])

        for param in self.model_audio.parameters():
            param.requires_grad = False
        for param in self.model_video.parameters():
            param.requires_grad = False
        param_groups = [{'params': self.model_fusion.parameters()}, {'params': self.criterion.parameters()}]
        if self.train_opts['optimizer'] == 'sgd':
            optim_opts = self.train_opts['sgd']
            self.optim = optim.SGD(param_groups, optim_opts['init_lr'], momentum = optim_opts['momentum'], weight_decay = optim_opts['weight_decay'])
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optim, milestones = self.train_opts['lr_decay_step'], gamma = 0.1)
        # print('criterion parameter:')
        # for param in self.criterion.parameters():
        #     print(param.requires_grad)
        #     print(param)

    ## embedding normalization
    def feature_normalize(self, data):
        mu = torch.mean(data,axis=1)
        std = torch.std(data,axis=1)
        a = data.transpose(0,1)
        b =  (a - mu)/std
        return b.transpose(0,1)

    ## batch normalization
    # def feature_normalize(self, data):
    #     mu = torch.mean(data,axis=0)
    #     std = torch.std(data,axis=0)
    #     return (data - mu)/std

    def extract_test_xv_lomgrid(self):
        parallel_model_audio = self.model_audio
        self.model_audio = self.model_audio.module
        self.model_audio.eval()

        parallel_model_video = self.model_video
        self.model_video = self.model_video.module
        self.model_video.eval()

        parallel_model_fusion = self.model_fusion
        self.model_fusion = self.model_fusion.module
        self.model_fusion.eval()

        print('Extracting test embeddings for LOMBARDGRID: ')
        # self._print_config(self.voxtestset.opts)
        os.makedirs('exp/test_em/test_em_lomgrid', exist_ok = True)
        with torch.no_grad():
            for input_video, input_audio, utt in tqdm(self.lomgridtestloader):
                utt = utt[0]
                input_audio = input_audio.to(self.device)
                # audio 
                xv_audio, _ = self.model_audio.extract_embedding(input_audio.to(self.device)) # lmcl use 2nd fc layer output as speaker embedding

                # video
                em_video = []
                for video_group in input_video:
                    em = 0
                    video_group = video_group.squeeze(0)
                    if len(video_group) > 0:
                        for v in video_group:
                            v = torch.FloatTensor(v)[None, None, :, :, :].to(self.device)
                            em += torch.mean(self.model_video(v, lengths=[v.shape[0]]).squeeze(-3),dim=0)
                        em_video.append(em / len(video_group))
                em_video = torch.stack(em_video)

                # features normalization
                # xv_audio = self.feature_normalize(xv_audio)
                # em_video = self.feature_normalize(em_video)

                # # feature fusion
                em = self.model_fusion(xv_audio, em_video)
                # em = self.feature_normalize(em)

                # # features normalization
                # xv_audio = self.feature_normalize(xv_audio)
                # em_video = self.feature_normalize(em_video)

                # # audio-visual fusion
                # # em = torch.cat([xv_audio,em_video,em], dim=1)
                # em = xv_audio + em_video + xv_audio * em_video
                # em = self.feature_normalize(em).squeeze(0)

                # em = em_video
                em = em.cpu().numpy()
                test_spk_dir = os.path.join('exp/test_em/test_em_lomgrid', os.path.dirname(utt))
                os.makedirs(test_spk_dir, exist_ok = True)
                np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), em)
        self.model_audio = parallel_model_audio
        self.model_video = parallel_model_video
        self.model_fusion = parallel_model_fusion

    def extract_test_xv_grid(self):
        parallel_model_audio = self.model_audio
        self.model_audio = self.model_audio.module
        self.model_audio.eval()

        parallel_model_video = self.model_video
        self.model_video = self.model_video.module
        self.model_video.eval()

        parallel_model_fusion = self.model_fusion
        self.model_fusion = self.model_fusion.module
        self.model_fusion.eval()

        print('Extracting test embeddings for GRID: ')
        # self._print_config(self.voxtestset.opts)
        os.makedirs('exp/test_em/test_em_grid', exist_ok = True)
        with torch.no_grad():
            for input_video, input_audio, utt in tqdm(self.gridtestloader):
                utt = utt[0]
                input_audio = input_audio.to(self.device)
                # audio 
                xv_audio, _ = self.model_audio.extract_embedding(input_audio.to(self.device)) # lmcl use 2nd fc layer output as speaker embedding

                # video
                em_video = []
                for video_group in input_video:
                    em = 0
                    video_group = video_group.squeeze(0)
                    if len(video_group) > 0:
                        for v in video_group:
                            v = torch.FloatTensor(v)[None, None, :, :, :].to(self.device)
                            em += torch.mean(self.model_video(v, lengths=[v.shape[0]]).squeeze(-3),dim=0)
                        em_video.append(em / len(video_group))
                em_video = torch.stack(em_video)

                # feature fusion
                em = self.model_fusion(xv_audio, em_video)
                # em = self.feature_normalize(em)

                # features normalization
                # xv_audio = self.feature_normalize(xv_audio)
                # em_video = self.feature_normalize(em_video)

                # audio-visual fusion
                # em = torch.cat([xv_audio,em_video,em], dim=1)
                # em = (xv_audio + em_video) / 2
                # em = self.feature_normalize(em).squeeze(0)
                                                                                                                                                                                                                                                                                                                                                        
                em = em.cpu().numpy()
                test_spk_dir = os.path.join('exp/test_em/test_em_grid', os.path.dirname(utt))
                os.makedirs(test_spk_dir, exist_ok = True)
                np.save(os.path.join(test_spk_dir, os.path.basename(utt).replace('.wav', '.npy')), em)
        self.model_audio = parallel_model_audio
        self.model_video = parallel_model_video
        self.model_fusion = parallel_model_fusion

    def _print_config(self, opts):
        pp = pprint.PrettyPrinter(indent = 2)
        pp.pprint(opts)

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
    mode = 'av_test'
    trainer = Trainer(mode)
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
        # trainer.extract_test_xv()
        # eer, threshold = utils.eer(trainer.log_time)
        # print("EER: {:.6f}%".format(eer * 100))
    # audio-visual dataset test
    elif mode == 'av_test':
        if trainer.test_opts['train_plda']:
            trainer.train_plda()
        if trainer.test_opts['eval_lomgrid']:
            trainer.extract_test_xv_lomgrid()
            if trainer.test_opts['use_cos']:              
                eer, threshold = utils.eer_cos_lomgrid('test_em')
                print("EER: {:.6f}%".format(eer * 100))
            if trainer.test_opts['use_plda']:              
                eer, threshold = utils.eer_plda_lomgrid('test_em')
                print("EER: {:.6f}%".format(eer * 100))
        if trainer.test_opts['eval_grid']:
            trainer.extract_test_xv_grid() 
            if trainer.test_opts['use_cos']:
                eer, threshold = utils.eer_cos_grid('test_em')
                print("EER: {:.6f}%".format(eer * 100))
            if trainer.test_opts['use_plda']:
                eer, threshold = utils.eer_plda_grid('test_em')
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

