import torch,math
from torch import nn, Tensor
from pretraining.track.transformers import Transformer
from einops import rearrange,repeat
from pretraining.track.layers import CNN,Enformer,AttentionPool
from einops.layers.torch import Rearrange

import numpy as np
import torch.nn.functional as F
class Tranmodel(nn.Module):
    def __init__(self, backbone, transfomer):
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, kernel_size=1)
    def forward(self, input):
        input=rearrange(input,'b n c l -> (b n) c l')
        src = self.backbone(input)
        src=self.input_proj(src)
        src = self.transformer(src)
        return src

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class finetunemodel(nn.Module):
    def __init__(self,pretrain_model,hidden_dim,embed_dim,bins,crop=50,num_class=245,return_embed=False):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.bins=bins
        self.crop=crop
        self.attention_pool = AttentionPool(hidden_dim)
        self.project=nn.Sequential(
            Rearrange('(b n) c -> b c n', n=bins),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7,groups=hidden_dim),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=9, padding=4),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.transformer = Enformer(dim=embed_dim, depth=4, heads=8)
        self.prediction_head=nn.Linear(embed_dim,num_class)
        self.return_embed=return_embed

    def forward(self, x):
        x=self.pretrain_model(x)
        x = self.attention_pool(x)
        x = self.project(x)
        x= rearrange(x,'b c n -> b n c')
        x = self.transformer(x)
        out = self.prediction_head(x[:, self.crop:-self.crop, :])
        if self.return_embed:
            return x
        else:
            return out


def build_backbone():
    model = CNN()
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
def build_track_model(args):
    backbone = build_backbone()
    transformer = build_transformer(args)
    pretrain_model = Tranmodel(
            backbone=backbone,
            transfomer=transformer,
        )

    if args.pretrain_path != 'none':
        print('load pre-training model: ' + args.pretrain_path)
        model_dict = pretrain_model.state_dict()
        pretrain_dict = torch.load(args.pretrain_path, map_location='cpu')
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        pretrain_model.load_state_dict(model_dict)

    if not args.fine_tune:
        for param in pretrain_model.parameters():
            param.requires_grad = False
    model=finetunemodel(pretrain_model,hidden_dim=args.hidden_dim,embed_dim=args.embed_dim,bins=args.bins,crop=args.crop,return_embed=args.return_embed)
    return model
