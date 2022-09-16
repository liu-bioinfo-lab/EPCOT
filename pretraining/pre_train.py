import os,sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from pretraining.model import build_model
import argparse
from torch.optim import lr_scheduler
import pickle, time
import random
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler

import argparse
from pretraining.layers import Balanced_AsymmetricLoss
from pretraining.dataset import Task1Dataset

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', default=245, type=int,help='number of labels')
    parser.add_argument('--seq_length', default=1600, type=int,help='input sequence length')
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--shuffle_dataset', default=True, action='store_false')
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--load_backbone', default=True, action='store_false', help='load trained backbones')
    parser.add_argument('--test', default=True, action='store_false', help='model testing')
    parser.add_argument('--ac_data', type=str, default='dnase_norm')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args


def data_loaders():
    args = get_args()
    dataset = Task1Dataset(ac_type=args.ac_data)
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    # train, validation split: 0.9,0.1
    valid_split = int(np.floor(dataset_size * 0.85))
    print(dataset_size, valid_split)
    if args.shuffle_dataset:
        print('shuffle datatset')
        np.random.seed(8)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[:valid_split], indices[valid_split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(dataset, batch_size=args.batchsize,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=args.batchsize,
                              sampler=valid_sampler)
    return train_loader,valid_loader




def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    model= build_model(args)
    model.to(device)
    criterion = Balanced_AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    cls = ['K562', 'MCF-7', 'GM12878', 'HepG2']

    train_loader, valid_loader=data_loaders()
    # mask unavailable epigenomic features for each cell line, prevent them from calculating loss
    with open('/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/pretrain_data/label_masks.pickle','rb') as f:
        labelmasks_pickle=pickle.load(f)
    labelmasks= torch.tensor(np.vstack([labelmasks_pickle[cl] for cl in cls])).float()

    best_loss=1e10
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
            if step % 1000== 0:
                print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "train_loss=",
                              "{:.7f}".format(cur_loss),
                              "time=", "{:.5f}".format(time.time() - t), "lr=", "{:.10f}".format(optimizer.param_groups[0]['lr'])
                              )
        scheduler.step()
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

        with open('pre_train_log_%s_%s.txt'%(args.backbone_type,args.ac_data),'a') as f:
            f.write('Epoch: %s, LR: %s, train_loss: %s, valid_loss: %s\n'%(epoch, optimizer.param_groups[0]['lr'], train_loss,valid_loss))
        if valid_loss < best_loss:
            print('save model')
            torch.save(model.state_dict(),
                       'models/pretrain_%s_%s.pt'%(args.backbone_type,args.ac_data))
            best_loss = valid_loss

if __name__=="__main__":
    main()