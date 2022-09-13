#most of the codes below are copied from Query2label
import torch,math
import numpy as np
from torch import nn, Tensor
from pretraining.layers import CNN
from pretraining.transformer import Transformer

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
        hidden_dim = transfomer.d_model
        self.label_input = torch.Tensor(np.arange(num_class)).view(1, -1).long()
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, input):
        src = self.backbone(input)
        label_inputs=self.label_input.repeat(src.size(0),1).cuda()
        label_embed=self.query_embed(label_inputs)
        src=self.input_proj(src)
        hs = self.transformer(src, label_embed)
        out = self.fc(hs)
        return out

def build_backbone(args):
    model = CNN()
    if args.load_backbone:
        # load trained backbone
        model_path='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/experiment/models/checkpoint1_cnn1_dnase_norm.pt'
        print(model_path)
        # model_path='models/backbone_%s.pt'%args.ac_data

        model_dict = model.state_dict()
        backbone_dict = torch.load(model_path, map_location='cpu')
        backbone_dict = {k: v for k, v in backbone_dict.items() if k in model_dict}
        model_dict.update(backbone_dict)
        model.load_state_dict(model_dict)
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
def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = Tranmodel(
            backbone=backbone,
            transfomer=transformer,
            num_class=args.num_class,
        )
    return model
