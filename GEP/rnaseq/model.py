import torch,math
import numpy as np
from torch import nn
from cage.layers import CNN
from cage.transformer import Transformer
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
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
        hs = self.transformer(src)
        return hs

class finetunemodel(nn.Module):
    def __init__(self,pretrain_model,hidden_dim,bins,finetune_model='lstm'):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.attention_pool1 = nn.Linear(hidden_dim, 1)
        self.bins=bins
        self.finetune_model=finetune_model
        if finetune_model == 'lstm':
            self.BiLSTM = nn.LSTM(input_size=hidden_dim, hidden_size=200, num_layers=2,
                                  batch_first=True,
                                  dropout=0.3,
                                  bidirectional=True)
            self.attention_pool2 = nn.Linear(400, 1)
            self.fc1 = nn.Sequential(
                nn.Linear(400, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        if finetune_model == 'cnn':
            self.cnn=nn.Sequential(
                nn.Conv1d(hidden_dim,360,kernel_size=1),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                Rearrange('b c l -> b l c'),
                nn.Conv1d(bins,128,kernel_size=20),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Conv1d(128, 128, kernel_size=10),
                nn.ReLU(),
            )
            self.fc3=nn.Linear(128,1)

    def forward(self, x):
        x=self.pretrain_model(x)
        x = torch.matmul(F.softmax(self.attention_pool1(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        if self.finetune_model=='lstm':
            x = rearrange(x, '(b n) c -> b n c', n=self.bins)
            x, (hn, cn)=self.BiLSTM(x)
            x=x.mean(1)
            x = self.fc1(x)
        else:
            x = rearrange(x, '(b n) c -> b c n', n=self.bins)
            x=self.cnn(x)
            x=x.mean(2)
            x=self.fc3(x)
        return x

def build_backbone(args):
    model = CNN(args.num_class, args.seq_length, args.embedsize)
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
def build_pre_train_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    pretrain_model = Tranmodel(
            backbone=backbone,
            transfomer=transformer,
            num_class=args.num_class,
        )
    if args.pretrain_path!='none':
        pretrain_model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
    if not args.fine_tune:
        for param in pretrain_model.parameters():
            param.requires_grad = False
    model=finetunemodel(pretrain_model=pretrain_model,hidden_dim=args.hidden_dim,bins=args.bins,finetune_model=args.finetune_model)
    return model


