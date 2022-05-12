import torch,math
import numpy as np
from torch import nn, Tensor,einsum
from .transformer import Transformer
from einops import rearrange
from .layers import AttentionPool,Enformer,CNN
from einops.layers.torch import Rearrange
import os
class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)
    def forward(self, x):
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class Tranmodel(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        feature_pos=backbone._n_channels
        hidden_dim = transfomer.d_model
        self.feature_pos_encoding = nn.Parameter(torch.randn(1, hidden_dim, feature_pos))
        self.label_input = torch.Tensor(np.arange(num_class)).view(1, -1).long()
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)
    def forward(self, input,fea_pos=False):
        input=rearrange(input,'b n c l -> (b n) c l')
        src = self.backbone(input)
        src=self.input_proj(src)
        if fea_pos:
            src+=self.feature_pos_encoding
        # src = src.permute(0, 2, 1)
        src = self.transformer(src)
        return src



class finetunemodel(nn.Module):
    def __init__(self,pretrain_model,hidden_dim,embed_dim,bins,crop=25,mode='lstm'):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.bins=bins
        self.crop=crop
        self.mode=mode
        self.attention_pool = AttentionPool(hidden_dim)
        self.project=nn.Sequential(
            Rearrange('(b n) c -> b c n', n=bins),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4,groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        if mode=='lstm':
            self.BiLSTM=nn.LSTM(input_size=embed_dim, hidden_size=embed_dim//2, num_layers=3,
                                     batch_first=True,
                                     dropout=0.2,
                                     bidirectional=True)
        elif mode == 'transformer':
            self.transformer = Enformer(dim=embed_dim, depth=4, heads=6)
        self.prediction_head=nn.Linear(embed_dim,1)
    def forward(self, x):
        x=self.pretrain_model(x)
        x = self.attention_pool(x)
        x = self.project(x)
        x= rearrange(x,'b c n -> b n c')
        if self.mode == 'lstm':
            x, (h_n, h_c) = self.BiLSTM(x)
        elif self.mode== 'transformer':
            x=self.transformer(x)
        x=self.prediction_head(x[:,self.crop:-self.crop,:])
        return x

def build_backbone(args):
    model = CNN(args.num_class, args.seq_length, args.rnn_embedsize)
    return model
def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers
    )
def build_pretrain_model_cage(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    pretrain_model = Tranmodel(
            backbone=backbone,
            transfomer=transformer,
            num_class=args.num_class,
        )
    if args.pretrain_path!=None:
        pretrain_model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
    if not args.fine_tune:
        for param in pretrain_model.parameters():
            param.requires_grad = False
    model=finetunemodel(pretrain_model,hidden_dim=args.hidden_dim,embed_dim=args.embed_dim,bins=args.bins,crop=args.crop,mode=args.mode)
    return model

