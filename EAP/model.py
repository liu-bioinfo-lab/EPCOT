import torch,math
import numpy as np
from torch import nn, Tensor,einsum
from typing import Optional, List
from EAP.layers import CNN,AttentionPool
from EAP.transformer import Transformer
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


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



class finetunemodel(nn.Module):
    def __init__(self,pretrain_model,hidden_dim,embed_dim,bins):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.bins=bins
        self.attention_pool = AttentionPool(hidden_dim)
        self.project=nn.Sequential(
            Rearrange('(b n) c -> b c n', n=bins),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=bins),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        )
        self.prediction_head=nn.Linear(embed_dim,1)
    def forward(self, x):
        x=self.pretrain_model(x)
        x = self.attention_pool(x)
        x = self.project(x)
        x=self.prediction_head(x.squeeze())
        return x


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


def build_pretrain_model_enhancer(args):
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
    model = finetunemodel(pretrain_model, hidden_dim=args.hidden_dim, embed_dim=args.embed_dim, bins=args.bins)
    return model
