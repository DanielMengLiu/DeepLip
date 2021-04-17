# -*- codingL utf-8 -*-

import numpy as np
import torch.nn as nn
import torch


class LowFER(nn.Module):
    def __init__(self, d1, d2, o):
        super(LowFER, self).__init__()
        k = 30
        self.U = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.V = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, k * o)),
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.input_dropout = nn.Dropout(0.3)
        self.hidden_dropout1 = nn.Dropout(0.4)
        self.hidden_dropout2 = nn.Dropout(0.5)
        self.bn0 = nn.BatchNorm1d(d1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.k = k
        self.o = o
    
    # def init(self):
    #     nn.init.xavier_normal_(self.E.weight.data)
    #     nn.init.xavier_normal_(self.R.weight.data)
    
    def forward(self, e1, e2):
        # e1 = nn.functional.normalize(e1, p=2, dim=-1)
        # e2 = nn.functional.normalize(e2, p=2, dim=-1)

        # e1 = self.bn0(e1)
        # e2 = self.bn0(e2)
        # e1 = self.input_dropout(e1)
        # e2 = self.input_dropout(e2)

        ## MFB
        x = torch.mm(e1, self.U) * torch.mm(e2, self.V)
        # x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.mean(-1)
        # x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = nn.functional.normalize(x, p=2, dim=-1)
        # x = self.bn1(x)
        # x = e1 + e2
        #x = nn.functional.normalize(x, p=2, dim=-1)
        # e1 = torch.relu(e1)
        e2 = torch.sigmoid(e2)
        x = e2 * e1
        x = torch.cat([e1,e2,x],dim=1)
        # x = self.hidden_dropout2(x)
        # x = torch.mm(x, self.E.weight.transpose(1, 0))
        # x = torch.sigmoid(x)
        return x

# class LowFER(nn.Module):
#     def __init__(self, M, N, O):
#         super(LowFER, self).__init__()
#         # self.input_dropout = nn.Dropout(0.3)
#         # self.hidden_dropout1 = nn.Dropout(0.4)
#         # self.hidden_dropout2 = nn.Dropout(0.5)
#         # self.bn0 = nn.BatchNorm1d(d1)
#         self.bn1 = nn.BatchNorm1d(O)
#         # self.activation = nn.LeakyReLU(negative_slope = 0.2)
#         self.fc1 = nn.Linear(M*N, O)
    
#     def forward(self, e1, e2):
#         # e1 = self.bn0(e1)
#         # e2 = self.bn0(e2)
#         # e1 = self.input_dropout(e1)
#         # e2 = self.input_dropout(e2)


#         ## MFB
#         # outer_product = torch.outer(e1, e2)
#         outer_product = torch.einsum('bi,bj->bij', e1, e2)
#         vector_outer = outer_product.view(outer_product.shape[0],-1)
#         x = self.fc1(vector_outer)

#         # x = self.activation(x)
#         # x = self.hidden_dropout1(x)
#         # x = x.view(-1, self.o, self.k)
#         # x = x.sum(-1)
#         # x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
#         x = nn.functional.normalize(x, p=2, dim=-1)
#         x = self.bn1(x)
#         x = x + e1 + e2
#         # x = self.hidden_dropout2(x)
#         # x = torch.mm(x, self.E.weight.transpose(1, 0))
        
#         # x = torch.sigmoid(x)
#         return x