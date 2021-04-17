import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from models.audio_models.pooling import *

class TDNN_Block(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            dilation,
            padding = 0,
            stride = 1,
            bn_first = True,
            ):
        super(TDNN_Block, self).__init__()
        kernel_size = len(dilation)
        if len(dilation) > 1:
            dilation = (dilation[-1] - dilation[0]) // (len(dilation) - 1)
        else:
            dilation = 1
        self.context_layer = nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation
                )
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        self.bn_first = bn_first

    def forward(self, x):
        x = self.context_layer(x)
        if self.bn_first:
            x = self.bn(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.bn(x)
        return x

class SpeakerEmbNet(nn.Module):
    def __init__(self, opts):
        super(SpeakerEmbNet, self).__init__()
        opts = opts[opts['arch']]
        context = opts['context']
        input_dim = opts['input_dim']
        hidden_dim = opts['hidden_dim']
        layers_num = opts['tdnn_layers']
        embedding_dim = opts['embedding_dim']
        attention_hidden_size = opts['attention_hidden_size']
        self.bn_first = opts['bn_first']
        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        layers = []

        for i in range(layers_num):
            layers.append(TDNN_Block(input_dim, hidden_dim[i], dilation = context[i], stride = 1, bn_first = self.bn_first))
            input_dim = hidden_dim[i]

        self.tdnn = nn.Sequential(*layers)

        # pooling method selection
        if opts['pooling'] == 'statistic':
            self.pooling = MeanStdPooling()
        elif opts['pooling'] == 'average':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif opts['pooling'] == 'attentive_statistic':
            self.pooling = AttentiveStatPooling(hidden_dim[-1], attention_hidden_size)
        elif opts['pooling'] == 'mono_head_attention':
            self.pooling = MonoHeadAttention(hidden_dim[-1], attention_hidden_size)
        else:
            raise NotImplementedError('Other pooling method has not implemented.')

        # first fc layer
        if opts['pooling'] == 'statistic' or opts['pooling'] == 'attentive_statistic':
            self.fc1 = nn.Linear(hidden_dim[-1] * 2, embedding_dim)
        elif opts['pooling'] == 'average' or opts['pooling'] == 'mono_head_attention':
            self.fc1 = nn.Linear(hidden_dim[-1], embedding_dim)
        else:
            raise ValueError("pooling method is wrong!")

        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def extract_embedding(self, x):
        x = self.tdnn(x)
        x = self.pooling(x)
        x.squeeze_(1)
        x_a = self.fc1(x)
        if self.bn_first:
            x = self.bn1(x_a)
            x = self.activation(x)
        else:
            x = self.activation(x_a)
            x = self.bn1(x)
        xv = self.fc2(x)
        return xv, x_a
        
    def forward(self, x):
        x, _ = self.extract_embedding(x)
        if self.bn_first:
            x = self.bn2(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.bn2(x)
        return x

if __name__ == '__main__':
    import yaml
    from torchsummary import summary
    import os
    f = open('./conf/config.yaml', 'r')
    opts = yaml.load(f, Loader = yaml.CLoader)
    f.close()
    print(type(opts['model']['tdnn']['context']))
    print(opts['model']['tdnn']['context'])
    net = SpeakerEmbNet(opts['model'])
    ckpt = torch.load('exp/Mon_Oct_19_22:17:11_2020/net_21.pth', map_location = 'cpu')
    state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if 'fc3' in k:
            continue
        state_dict[k.replace('module.', '')] = v
    #  for name, value in net.named_parameters():
    #      print(name)
    net.load_state_dict(state_dict)

    print(net)
    #  summary(net.cuda(), (24, 100))
