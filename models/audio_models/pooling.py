import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention


class MeanStdPooling(torch.nn.Module):
    """
    Mean and Standard deviation pooling
    """
    def __init__(self):
        """

        """
        super(MeanStdPooling, self).__init__()
        pass

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)

class MonoHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MonoHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation = nn.ReLU()
        self.W = nn.Parameter(torch.Tensor(1, hidden_size, input_size)).cuda()
        self.b = nn.Parameter(torch.Tensor(1, hidden_size, 1)).cuda()
        self.v = nn.Parameter(torch.Tensor(1, 1, hidden_size)).cuda()
        self.k = nn.Parameter(torch.Tensor(1, 1, 1)).cuda()
        self._initialize_parameters()

    def _initialize_parameters(self):
        for parameter in self.parameters():
            nn.init.xavier_normal_(parameter)
            
    def get_attention(self, x):
        '''
        params:
            x: (C, T)
        return:
            alpha: (1, T)
        '''
        hidden_mat = self.W.matmul(x) + self.b
        x = self.activation(hidden_mat)
        e = self.v.matmul(hidden_mat) + self.k
        alpha = F.softmax(e, dim = 2)
        return alpha 
    
    def forward(self, x):
        alpha = self.get_attention(x)
        attention_mean = alpha.matmul(x.transpose(1, 2))
        return attention_mean

# TODO
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, head = 4, hidden_size = [100,200,300,400]):
        super(MultiHeadAttention, self).__init__()

    def get_attention(self):
        pass
    
    def forward(self, x):
        pass

class AttentiveStatPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentiveStatPooling, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation = nn.ReLU()
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b = nn.Parameter(torch.Tensor(1, hidden_size))
        self.v = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.k = nn.Parameter(torch.Tensor(1, 1))
        self._initialize_parameters()

    def _initialize_parameters(self):
        for parameter in self.parameters():
            nn.init.xavier_normal_(parameter)
        
    def get_attention(self, x):
        '''
        params:
            x: (B, C, T)
        return:
            alpha: (1, T)
        '''
        hidden_mat = self.W.matmul(x).transpose(1, 2) + self.b # B, T, H
        x = self.activation(hidden_mat)
        e = x.matmul(self.v) + self.k
        alpha = F.softmax(e, dim = 1)
        return alpha
        
    def forward(self, x):
        alpha = self.get_attention(x)
        attention_mean = torch.matmul(x, alpha).squeeze()
        attention_std = torch.sqrt(torch.matmul(x * x, alpha).squeeze() - attention_mean * attention_mean)
        attention_embedding = torch.cat([attention_mean, attention_std], dim = 1)
        return attention_embedding

if __name__ == '__main__':
    attentive_pooling = AttentiveStatPooling(512, 64)
    inputs = torch.randn(32, 512, 100)
    spk_emb = attentive_pooling(inputs)
    print(spk_emb.shape)
    
