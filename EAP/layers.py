import torch
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn, einsum
class CNN(nn.Module):
    def __init__(self, nclass, seq_length,embed_length):
        super(CNN, self).__init__()
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4
        sequence_length = seq_length
        n_targets = nclass
        linear_size = embed_length
        self.conv_net = nn.Sequential(
            nn.Conv1d(5, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size1, stride=pool_kernel_size1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2, stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2))
        reduce_by1 = 2 * (conv_kernel_size1 - 1)
        reduce_by2 = 2 * (conv_kernel_size2 - 1)
        pool_kernel_size1 = float(pool_kernel_size1)
        pool_kernel_size2 = float(pool_kernel_size2)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by1) / pool_kernel_size1)
                 - reduce_by2) / pool_kernel_size2)
            - reduce_by2)
        self.linear = nn.Linear(512* self._n_channels, linear_size)
        self.batch_norm = nn.BatchNorm1d(linear_size)
        self.classifier = nn.Linear(linear_size, n_targets)
        self.relu = nn.ReLU()
        self.num_channels = 512
    def forward(self, x):
        out = self.conv_net(x)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_fn = Rearrange('b (n p) d-> b n p d', n=1)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        attn_logits = einsum('b n d, d e -> b n e', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -2)
        return (x * attn).sum(dim = -2).squeeze()
