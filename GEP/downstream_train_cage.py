from cage.util import prepare_train_data
from cage.model import build_pretrain_model_cage
import argparse
from torch.optim import lr_scheduler
import os, pickle, time
import random
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
import argparse
from cage_dataset import inputDataset,clDataset
from scipy.stats import pearsonr,spearmanr
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=250)
    parser.add_argument('--crop', type=int, default=25)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--embed_dim', default=360, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--accum_iter', default=16, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--shuffle_dataset', default=True, action='store_false', help='model testing')
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--test', default=True, action='store_false', help='model testing')
    parser.add_argument('--ac_data', type=str, default='dnase_norm')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--mode', type=str, default='transformer',choices=['transformer','lstm'])
    parser.add_argument('--pretrain_path', type=str, default='none',help='path to the saved pre-training model')
    # parser.add_argument('--pretrain_path',type=str,default='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/pretrain/models/checkpoint_pretrain_cnn1_dnase_norm.pt')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args

def load_data(data,dnase_data, ref_data,chroms,all_cls):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data=data.numpy().astype(int)
    input=[]
    for i in range(data.shape[0]):
        chr=chroms[data[i][0]]
        cl=all_cls[data[i][-1]]
        s,e=data[i][1]//1000,data[i][2]//1000
        input.append(torch.cat((ref_data[chr][s:e],dnase_data[cl][chr][s:e]),dim=1))
    input= torch.stack(input)
    return input.float().to(device)

def shuffle_data(dataset_size,seed=8):
    # randomly split training/validation/testing sets
    indices=np.arange(dataset_size)
    valid_split=int(np.floor(dataset_size*0.8))
    test_split=int(np.floor(dataset_size*0.9))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return indices[:valid_split],indices[valid_split:test_split],indices[test_split:]

def generate_label(cl_index,indices,all_cls,cage_data):
    labels=[]
    for idx in cl_index:
        cl=all_cls[idx]
        temp=cage_data[cl][indices]
        labels.append(temp)
    return np.vstack(labels)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    model= build_pretrain_model_cage(args)
    model.cuda()
    criterion= torch.nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=1e-6)

    all_cls=['GM12878','K562','HUVEC','IMR-90','H1','A549','HepG2','CD14+']
    traincl_index=[0,1,2,3]
    testcl_index=[4,5,6,7]
    chroms=[str(i) for i in range(1,23)]

    # all_cls=['CD14+CD25+','spleen','thymus','lung','cerebellum']
    # traincl_index=[0,1,2]
    # testcl_index=[3,4]
    # chroms=[i for i in range(1,20)]

    # load input regions
    # input_locs = np.load('/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/gene_exp/ct_cage/data/250kb/input_region_mm.npy')
    input_locs=np.load('cage/input_region.npy')

    dnase_data, ref_data,cage_data=prepare_train_data(all_cls,chroms)


    train_index,valid_index,test_index=shuffle_data(dataset_size=input_locs.shape[0])
    train_dataset=inputDataset(input_locs[train_index],generate_label(traincl_index,train_index,all_cls,cage_data),traincl_index)
    train_loader=DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True)
    valid_dataset={}
    valid_loader={}
    test_dataset={}
    test_loader={}
    for cidx in traincl_index:
        valid_dataset[cidx]=clDataset(input_locs[valid_index],generate_label([cidx],valid_index,all_cls,cage_data),cidx)
        valid_loader[cidx]=DataLoader(valid_dataset[cidx],batch_size=args.batchsize,shuffle=False)
        test_dataset[cidx]=clDataset(input_locs[test_index],generate_label([cidx],test_index,all_cls,cage_data),cidx)
        test_loader[cidx]=DataLoader(test_dataset[cidx],batch_size=args.batchsize,shuffle=False)
    testct_dataset={}
    testct_loader={}
    for cidx in testcl_index:
        testct_dataset[cidx]=clDataset(input_locs[test_index],generate_label([cidx],test_index,all_cls,cage_data),cidx)
        testct_loader[cidx]=DataLoader(testct_dataset[cidx],batch_size=args.batchsize,shuffle=False)

    if args.mode=='transformer' and not args.load_model:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,steps_per_epoch=len(train_loader),
                        epochs=args.epochs, pct_start=0.025, div_factor=100,final_div_factor=0.01)
    best_criter=0
    for epoch in range(args.epochs):
        training_losses=[]
        model.train()
        for step, (input_indices,input_label) in enumerate(train_loader):

            t=time.time()
            input_label=input_label.float().to(device)
            input_data=load_data(input_indices,dnase_data, ref_data,chroms,all_cls)
            output = model(input_data)
            loss = criterion(output, input_label)
            loss = loss / args.accum_iter
            loss.backward()
            if ((step + 1) % args.accum_iter == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            if args.mode == 'transformer' and not args.load_model:
                scheduler.step()
            cur_loss = loss.item()
            training_losses.append(cur_loss)
            if step % 2000 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "train_loss=",
                          "{:.7f}".format(cur_loss),
                          "time=", "{:.5f}".format(time.time() - t), "lr=", "{:.10f}".format(optimizer.param_groups[0]['lr'])
                          )

        train_loss = np.average(training_losses)
        print('Epoch: {} LR: {:.8f} train_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], train_loss))
        model.eval()
        pccs = []
        for cidx in traincl_index:
            validation_losses = []
            pred_eval = []
            target_eval = []

            for step, (input_indices,input_label) in enumerate(valid_loader[cidx]):
                t = time.time()
                with torch.no_grad():
                    input_label = input_label.float().to(device)
                    input_data = load_data(input_indices,dnase_data, ref_data,chroms,all_cls)

                    output = model(input_data)
                    loss = criterion(output, input_label)

                    cur_loss = loss.item()
                    validation_losses.append(cur_loss)

                    pred_eval.append(output.cpu().data.detach().numpy())
                    target_eval.append(input_label.cpu().data.detach().numpy())
            pred_eval = np.concatenate(pred_eval, axis=0).squeeze()
            target_eval = np.concatenate(target_eval, axis=0).squeeze()
            pcc, _ = pearsonr(pred_eval.flatten(), target_eval.flatten())
            pccs.append(pcc)
        criter=np.mean(pccs)

        if criter > best_criter:
            best_criter=criter
            print('save model')
            torch.save(model.state_dict(),
                       'models/%s_%s_%s.pt' % (args.mode,args.bins, args.ac_data))

            if args.test:
                pccs=[]
                model.eval()
                for m in model.modules():
                    if m.__class__.__name__.startswith('BatchNorm'):
                        m.train()
                for cidx in testcl_index:
                    pred_eval = []
                    target_eval = []
                    with torch.no_grad():
                        for step, (input_indices,input_label) in enumerate(testct_loader[cidx]):
                            t = time.time()
                            input_label = input_label.float().to(device)
                            input_data= load_data(input_indices,dnase_data, ref_data,chroms,all_cls)
                            output = model(input_data)
                            pred_eval.append(output.cpu().data.detach().numpy())
                            target_eval.append(input_label.cpu().data.detach().numpy())
                    pred_eval = np.concatenate(pred_eval, axis=0).squeeze()
                    target_eval = np.concatenate(target_eval, axis=0).squeeze()
                    pcc, _ = pearsonr(pred_eval.flatten(), target_eval.flatten())
                    pccs.append(pcc)

if __name__=="__main__":
    main()