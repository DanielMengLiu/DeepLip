#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from lipreading.utils import load_json, save2npz
from lipreading.model import Lipreading
from lipreading.dataloaders import get_train_data_loaders, get_data_loaders, get_preprocessing_pipelines
from torchsummary import summary
import time
import logging
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from torch.nn import Parameter


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='tcdtimit', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=62, help='Number of classes')
    # -- directory
    parser.add_argument('--data-dir', default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/datasets_TCDTIMIT_1/', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/labels/57SpeakerLabel.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type = str, default = 'relu', choices = ['relu','prelu'], help = 'what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--maxepoch', type=float, default=300, help='max epochs')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--batch-size', type=int, default=45, help='Mini-batch size')
    # -- test
    parser.add_argument('--model-path', type=str, default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/models/lrw_resnet18_mstcn.pth.tar', help='Pretrained model pathname')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/configs/lrw_resnet18_mstcn.json', help='Model configuration with json format')
    # -- output
    parser.add_argument('--display', type=int, default=100, help='iteration display')
    parser.add_argument('--save-path', type=str, default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/models/epoch', help='model savepath')
    parser.add_argument('--log-path', type=str, default='/data/liumeng/Lipreading_using_Temporal_Convolutional_Networks/logs', help='logs savepath')

    args = parser.parse_args()
    return args

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device_ids=range(torch.cuda.device_count())

writer = SummaryWriter()
args = load_args()

# logging info
filename = args.log_path + '/lr_' + str(args.lr) + 'tcdtimit.txt'
logger_name = "log"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename, mode='a')
fh.setLevel(logging.INFO)
logger.addHandler(fh)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])

def train(model, dset_loader):
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=4e-08)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.maxepoch):

        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.maxepoch - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

        model.train()

        running_loss, running_corrects, running_all = 0., 0., 0.

        for batch_idx, (inputs, lengths, labels) in enumerate(dset_loader):
            inputs = Variable(inputs.unsqueeze(1).cuda())
            labels = Variable(labels.cuda())
            lengths = Variable(torch.from_numpy(np.array(lengths)).cuda())

            out = model(inputs, lengths=lengths)
            # ===========================================
            # F_test = Parameter(torch.Tensor(57, 500)).cuda()
            # nn.init.xavier_normal_(F_test)
            # out = F.linear(out, F_test)
            # F_test2 = Parameter(torch.Tensor(57, 256)).cuda()
            # nn.init.xavier_normal_(F_test2)
            # out = F.linear(out, F_test2)
            # ===========================================
            loss = criterion(out, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()      
                  
            _, preds = torch.max(F.softmax(out,dim=1), 1)     
            
            running_loss += loss.data * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_all += len(inputs)

            if batch_idx == 0:
                since = time.time()
            elif batch_idx % args.display == 0 or (batch_idx == len(dset_loader)-1):
                print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}%\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    len(dset_loader.dataset),
                    100. * batch_idx / (len(dset_loader)-1),
                    running_loss / running_all,
                    100. * running_corrects / running_all,
                    time.time()-since,
                    (time.time()-since)*(len(dset_loader)-1) / batch_idx - (time.time()-since)))

        logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
            'train',
            epoch,
            running_loss / len(dset_loader.dataset),
            running_corrects / len(dset_loader.dataset))+'\n')

        torch.save(model.state_dict(), args.save_path+'/'+str(epoch+1)+'.pt')



def get_model():
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                  }

    return Lipreading( num_classes=args.num_classes,
                       tcn_options=tcn_options,
                       backbone_type=args.backbone_type,
                       relu_type=args.relu_type,
                       width_mult=args.width_mult,
                       extract_feats=args.extract_feats).cuda()


def main():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        "'.json' config path does not exist. Path input: {}".format(args.config_path)
    '''
    assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
        "'.tar' model path does not exist. Path input: {}".format(args.model_path)
    '''
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.isdir(args.log_path):
        os.mkdir(args.log_path)

    model = get_model()
    if len(device_ids)>1:
        model=torch.nn.DataParallel(model)

    #model.load_state_dict(torch.load(args.model_path)["model_state_dict"], strict=True)

    if args.mouth_patch_path:
        save2npz( args.mouth_embedding_out_path, data = extract_feats(model).cpu().detach().numpy())
        return

    # -- get training dataset iterators
    train_dset_loaders = get_train_data_loaders(args)
    train(model, train_dset_loaders['train']) 

    # -- get dataset iterators
    #dset_loaders = get_data_loaders(args)
    #evaluate(model, dset_loaders['test'])

if __name__ == '__main__':
    main()