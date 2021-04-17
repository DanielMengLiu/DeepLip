# coding: utf-8
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


class Linearfusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, extract_feats):
        super(Linearfusion, self).__init__()
        self.extract_feats = extract_feats
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.LeakyReLU(negative_slope = 0.2)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        out = self.fc2(x1)
        return x1 if self.extract_feats else out

def model_fusion(input_size, hidden_size, num_classes, extract_feats):
    model = Linearfusion(input_size, hidden_size, num_classes, extract_feats)
    return model