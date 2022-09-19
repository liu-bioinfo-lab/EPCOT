### cross-cell type Hi-C contact map inference

from util import prepare_train_data
from hic.model import build_pretrain_model_hic
import os, pickle, time
import random
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
# from sklearn import metrics
import argparse
from hic_dataset import hic_dataset
from scipy.stats import pearsonr,spearmanr

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=200)
    parser.add_argument('--crop', type=int, default=4)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--shuffle_dataset', default=True, action='store_false', help='model testing')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--test', default=True, action='store_false', help='model testing')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--ac_data', type=str, default='dnase_norm')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    parser.add_argument('--trunk', type=str, default='transformer')
    parser.add_argument('--pretrain_path', type=str, default='none', help='path to the pre-training model')
    # parser.add_argument('--pretrain_path',type=str,default='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/pretrain/models/checkpoint_pretrain_cnn1_dnase_norm.pt')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args

def load_hic_model(
        saved_model_path,
        device
):
    """
    Load EPCOT model to predict 5kb-resolution Hi-C or ChIA-pet contact maps.
    """
    hic_args=get_args()
    hic_model = build_pretrain_model_hic(hic_args)
    model_dict = hic_model.state_dict()
    downstream_dict = torch.load(saved_model_path, map_location='cpu')
    downstream_dict = {k: v for k, v in downstream_dict.items() if k in model_dict}
    model_dict.update(downstream_dict)
    hic_model.load_state_dict(model_dict)
    hic_model.eval()
    hic_model.to(device)
    return hic_model

def upper_tri(x):
    args=get_args()
    effective_lens = args.bins - 2 * args.crop
    triu_tup = np.triu_indices(effective_lens)
    array_indices = np.array(list(triu_tup[1] + effective_lens * triu_tup[0]))
    return x.reshape(-1,effective_lens**2,1)[:,array_indices, :]

def complete_mat(mat):
    temp = mat.copy()
    np.fill_diagonal(temp,0)
    mat= mat+temp.T
    return mat

def arraytouptri(arrays):
    args = get_args()
    effective_lens = args.bins - 2 * args.crop
    triu_tup = np.triu_indices(effective_lens)
    temp=np.zeros((effective_lens,effective_lens))
    temp[triu_tup]=arrays
    return temp
def main(
        test_cl='IMR-90',
        cross_cell_type=True
):
    """
    Args:
        test_cl (str): the name of a cell/tissue type whose Hi-C is to predicted
        cross_cell_type (bool): if performing cross-cell type prediction
    """
    args=get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    saved_GM12878_model='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/5kb/models/GM12878_transformer.pt'
    saved_HFF_model='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/5kb/models/HFFc6_transformer.pt'
    hic_model_gm=load_hic_model(saved_GM12878_model,device)
    hic_model_hff = load_hic_model(saved_HFF_model, device)
    if cross_cell_type:
        for m in hic_model_gm.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                m.train()
        for m in hic_model_hff.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                m.train()

    chroms = [str(i) for i in range(1, 23)]
    test_chrs = ['3', '11', '17']

    dnase_data, ref_data, hic_data = prepare_train_data(test_cl, test_chrs)

    def load_data(data):
        data = data.numpy().astype(int)
        input = []
        label = []
        for i in range(data.shape[0]):
            chr = chroms[data[i][0]]
            s, e = data[i][1], data[i][2]
            input.append(torch.cat((ref_data[chr][s:e], dnase_data[chr][s:e]), dim=1))

            s, e = s // 5, e // 5
            temp = hic_data[chr][s + args.crop:e - args.crop, s + args.crop:e - args.crop].toarray()
            label.append(temp)
        input = torch.stack(input)
        label = torch.tensor(upper_tri(np.stack(label)))
        return input.float().to(device), label.float().to(device)

    testdataset = hic_dataset(test_chrs)
    test_loader = DataLoader(testdataset, batch_size=args.batchsize, shuffle=False)

    target_upi = []
    pred_upi = []
    scores = {'pcc': [], 'spm': []}
    with torch.no_grad():
        for step, input_indices in enumerate(test_loader):
            t = time.time()
            input_data, input_label = load_data(input_indices)
            if torch.sum(input_label) == 0:
                print(step, input_indices)
                continue
            output = hic_model_gm(input_data)
            output1 = hic_model_hff(input_data)
            out_array = output.cpu().data.detach().numpy().flatten()
            out_array1 = output1.cpu().data.detach().numpy().flatten()
            out_array = (out_array + out_array1) / 2

            target_array = input_label.cpu().data.detach().numpy().flatten()

            target_upi.append(arraytouptri(target_array.squeeze()))
            pred_upi.append(arraytouptri(out_array.squeeze()))

            pcc, _ = pearsonr(out_array.squeeze(), target_array.squeeze())
            spm, _ = spearmanr(out_array.squeeze(), target_array.squeeze())
            print(pcc)
            scores['pcc'].append(pcc)
            scores['spm'].append(spm)

    print(np.mean(scores['pcc']),np.mean(scores['spm']))


if __name__=='__main__':
    main(test_cl='IMR-90')



