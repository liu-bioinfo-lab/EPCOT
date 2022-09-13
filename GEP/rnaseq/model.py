import torch,math
import numpy as np
from torch import nn
from cage.layers import CNN
from cage.transformer import Transformer
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from cage.model import build_backbone,build_transformer,Tranmodel
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

def build_pre_train_model(args):
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
    model=finetunemodel(pretrain_model=pretrain_model,hidden_dim=args.hidden_dim,bins=args.bins,finetune_model=args.finetune_model)
    return model


