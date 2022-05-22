import os, pickle, time,sys
import random
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
from scipy.sparse import load_npz
import torch.optim as optim
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler

import argparse
from pretraining.layers import backbone_CNN,Balanced_AsymmetricLoss
from train_util import best_param
from dataset import Task1Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--num_class', default=245, type=int,help='number of labels')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--load_model', default=False, action='store_true', help='load trained model')
parser.add_argument('--test', default=True, action='store_false', help='model testing')
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--shuffle_dataset', default=True, action='store_false')
parser.add_argument('--ac_data', type=str, default='dnase_norm')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model=backbone_CNN(nclass=args.num_class, seq_length=1600, embed_length=320)
model.to(device)
criterion = Balanced_AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

dataset=Task1Dataset(ac_type=args.ac_data)
dataset_size=len(dataset)
indices=np.arange(dataset_size)
# train, validation split: 0.9,0.1
valid_split=int(np.floor(dataset_size*0.9))
if args.shuffle_dataset:
    np.random.seed(8)
    np.random.shuffle(indices)
train_indices,valid_indices=indices[:valid_split],indices[valid_split:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
train_loader = DataLoader(dataset, batch_size=args.batchsize,
                                           sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=args.batchsize,
                                                sampler=valid_sampler)


cls=['K562','MCF-7','GM12878','HepG2']
with open('/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/pretrain_data/label_masks.pickle','rb') as f:
    labelmasks_pickle=pickle.load(f)

labelmasks= torch.tensor(np.vstack([labelmasks_pickle[cl] for cl in cls])).float()
best_loss=1000
for epoch in range(args.epochs):
    training_losses=[]
    model.train()
    for step, (input_seq,input_dnase,input_label,label_mask) in enumerate(train_loader):
        t = time.time()
        train_seq= torch.FloatTensor(torch.cat((input_seq,input_dnase),1)).to(device)
        train_lmask= labelmasks[label_mask]\
            .view(input_seq.shape[0],args.num_class).to(device)
        train_target = input_label.float().to(device)
        output = model(train_seq)
        loss = criterion(output, train_target, train_lmask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        training_losses.append(cur_loss)
        if step % 10000 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "train_loss=",
                          "{:.7f}".format(cur_loss),
                          "time=", "{:.5f}".format(time.time() - t),
                          )

    validation_losses = []
    model.eval()
    for step, (input_seq, input_dnase, input_label, label_mask) in enumerate(valid_loader):
        t = time.time()
        valid_seq = torch.FloatTensor(torch.cat((input_seq, input_dnase), 1)).to(device)
        valid_lmask = labelmasks[label_mask] \
            .view(input_seq.shape[0], args.num_class).to(device)
        valid_target = input_label.float().to(device)
        with torch.no_grad():
            output = model(valid_seq)
            loss = criterion(output, valid_target, valid_lmask)
            cur_loss = loss.item()
            validation_losses.append(cur_loss)
            if step % 10000 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "validation_loss=",
                              "{:.7f}".format(cur_loss),
                              "time=", "{:.5f}".format(time.time() - t)
                              )
    train_loss = np.average(training_losses)
    print('Epoch: {} LR: {:.8f} train_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], train_loss))
    valid_loss = np.average(validation_losses)
    print('Epoch: {} LR: {:.8f} valid_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], valid_loss))
    with open('backbone_%s_log.txt'%(args.ac_data), 'a') as f:
        f.write('Epoch: %s, train loss: %s, valid_loss: %s\n' % (epoch,train_loss,valid_loss))
    if valid_loss< best_loss:
        print('save model')
        torch.save(model.state_dict(),'models/backbone_%s.pt'%(args.ac_data))