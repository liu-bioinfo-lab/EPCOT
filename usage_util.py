#functions used in

import kipoiseq
from kipoiseq import Interval
import numpy as np
import torch


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)
def pad_seq_matrix(matrix, pad_left, pad_right, pad_len=300):
    # add flanking region to each sample
    dmatrix = np.concatenate((pad_left, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], pad_right), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)


def pad_signal_matrix(matrix, pad_len=300):
    paddings = np.zeros(pad_len).astype('float32')
    dmatrix = np.vstack((paddings, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], paddings))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))


def generate_input(fasta_extractor,chrom, start, end, dnase):
    if start>=end:
        raise ValueError('the start of genomic region should be small than the end.')
    target_interval = kipoiseq.Interval(chrom, start, end)
    sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval))
    sequence_matrix = sequence_one_hot.reshape(-1, 1000, 4).swapaxes(1, 2)

    pad_interval = kipoiseq.Interval(chrom, start - 300, start)
    seq_pad_left = np.expand_dims(one_hot_encode(fasta_extractor.extract(pad_interval)).swapaxes(0, 1), 0)

    pad_interval = kipoiseq.Interval(chrom, end, end + 300)
    seq_pad_right = np.expand_dims(one_hot_encode(fasta_extractor.extract(pad_interval)).swapaxes(0, 1), 0)
    dnase_interest = np.expand_dims(pad_signal_matrix(dnase.reshape(-1, 1000))[start // 1000:end // 1000], 1)
    seq_interest = pad_seq_matrix(sequence_matrix, seq_pad_left, seq_pad_right)
    inputs = torch.tensor(np.concatenate((seq_interest, dnase_interest), axis=1)).float()
    return inputs

def plot_atac(ax,val,color='#17becf'):
    ax.fill_between(np.arange(val.shape[0]), 0, val, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.margins(x=0)
    ax.set_xticks([])

def plot_bindings(ax, val, chr, start, end, color='#17becf'):
    ax.fill_between(np.arange(val.shape[0]), 0, val, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks(np.arange(0, val.shape[0], val.shape[0] // 5))
    ax.set_ylim(0, 1)
    ax.set_xticklabels(np.arange(start, end, (end - start) // 5))
    ax.margins(x=0)

def plot_cage(ax,val,chr,start,end,color='#17becf'):
    ax.fill_between(np.arange(val.shape[0]), 0, val, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel('%s:%s-%s'%(chr,start,end))
    ax.set_xticks(np.arange(0,val.shape[0],val.shape[0]//5))
    ax.set_xticklabels(np.arange(start,end,(end-start)//5))
    ax.margins(x=0)

def predict_epis(model,chrom, start,end,dnase,fasta_extractor,device):
    if (end-start)%1000:
        raise ValueError('the length of the input genomic region should be divisible by 1000')
    model.eval()
    inputs = generate_input(fasta_extractor,chrom, start, end, dnase).to(device)
    with torch.no_grad():
        pred_epi = torch.sigmoid(model(inputs))
    pred_epi = pred_epi.detach().cpu().numpy()
    return pred_epi

def predict_cage(model,chrom,start,end,dnase,fasta_extractor,device):
    if (end-start)%200000:
        raise ValueError('the length of the input genomic region should be divisible by 200000')
    model.eval()
    input_start, input_end = start - 25000, end + 25000
    inputs = []
    for s in range(input_start, input_end - 200000, 200000):
        e = s + 250000
        cage_inputs = generate_input(fasta_extractor,chrom, s, e, dnase).to(device)
        inputs.append(cage_inputs)
    inputs = torch.stack(inputs)
    with torch.no_grad():
        pred_cage = model(inputs).detach().cpu().numpy().flatten()
    return pred_cage

def arraytouptri(arrays,args):
    effective_lens=args.bins-2*args.crop
    triu_tup = np.triu_indices(effective_lens)
    temp=np.zeros((effective_lens,effective_lens))
    temp[triu_tup]=arrays
    return temp
def complete_mat(mat):
    temp = mat.copy()
    np.fill_diagonal(temp,0)
    mat= mat+temp.T
    return mat
def plot_hic(ax, mat,cmap='RdBu_r', vmin=0, vmax=5):
    ax.imshow(mat,cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

def predict_hic(model,chrom,start,end,dnase,fasta_extractor,device):
    if (end-start)%1000000:
        raise ValueError('the length of the input genomic region should be divisible by 1000000')
    inputs = []
    for s in np.arange(start, end, 1000000):
        inputs.append(generate_input(fasta_extractor,chrom, s, s + 1000000, dnase))
    inputs = torch.stack(inputs).float().to(device)
    with torch.no_grad():
        pred_hic = model(inputs).detach().cpu().numpy()
    return pred_hic


def predict_microc(model,chrom,start,end,dnase,fasta_extractor,device):
    if (end-start)%600000:
        raise ValueError('the length of the input genomic region should be divisible by 600000')
    inputs = []
    for s in np.arange(start, end, 600000):
        inputs.append(generate_input(fasta_extractor,chrom, s, s + 600000, dnase))
    inputs = torch.stack(inputs).float().to(device)
    with torch.no_grad():
        pred_hic = model(inputs).detach().cpu().numpy()
    return pred_hic