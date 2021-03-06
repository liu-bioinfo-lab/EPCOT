#functions used in
import argparse
import kipoiseq
from kipoiseq import Interval
import numpy as np
import torch,pickle
from pretraining.model import build_model
from GEP.cage.model import build_pretrain_model_cage
from COP.hic.model import build_pretrain_model_hic
from COP.microc.model import build_pretrain_model_microc


def load_input_dnase(pickle_file):
    with open(pickle_file,'rb') as f:
        dnase=pickle.load(f)
    return dnase

def read_dnase_pickle(dnase_in_pickle,chrom):
    try:
        chrom_keys = dnase_in_pickle.keys()
        if chrom not in chrom_keys:
            if isinstance(chrom,str):
                chrom=int(chrom)
        input_dnase=dnase_in_pickle[chrom].toarray().squeeze()
    except Exception:
        raise ValueError('Please enter a correct chromosome')
    return input_dnase

def search_tf(tf):
    with open('EPCOT/Profiles/epigenomes.txt', 'r') as f:
        epigenomes = f.read().splitlines()
    try:
        tf_idx= epigenomes.index(tf)
    except Exception:
        raise ValueError("please specify a TF in the list of predicted TFs")
    return tf_idx


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)
def pad_seq_matrix(matrix, pad_left, pad_right, pad_len=300):
    # add flanking region to each sample
    dmatrix = np.concatenate((pad_left, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], pad_right), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)


def pad_signal_matrix(matrix,pad_dnase_left,pad_dnase_right, pad_len=300):
    dmatrix = np.vstack((pad_dnase_left, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], pad_dnase_right))[1:, :]
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
    seq_input = pad_seq_matrix(sequence_matrix, seq_pad_left, seq_pad_right)

    pad_dnase_left=dnase[start-300:start]
    pad_dnase_right = dnase[end:end+300]
    dnase_input = np.expand_dims(pad_signal_matrix(dnase[start:end].reshape(-1, 1000),pad_dnase_left,pad_dnase_right), 1)

    inputs = torch.tensor(np.concatenate((seq_input, dnase_input), axis=1)).float()
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

### arguments for pre-training model
def parser_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_class', default=245, type=int)
    parser.add_argument('--seq_length', default=1600, type=int)
    parser.add_argument('--embedsize', default=320, type=int)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--fea_pos', default=False, action='store_true')
    parser.add_argument('--load_backbone', default=False)
    args, unknown = parser.parse_known_args()
    return args,parser
def get_args():
    args,_ = parser_args()
    return args,_
### arguments for downstream model to predict 1kb-resolution CAGE-seq
def parser_args_cage(parent_parser):
    parser=argparse.ArgumentParser(parents=[parent_parser])
    parser.add_argument('--bins', type=int, default=250)
    parser.add_argument('--crop', type=int, default=25)
    parser.add_argument('--pretrain_path', type=str, default='none')
    parser.add_argument('--embed_dim', default=360, type=int)
    parser.add_argument('--mode', type=str, default='transformer')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    args, unknown = parser.parse_known_args()
    return args
### arguments for downstream model to predict 5kb-resolution Hi-C and ChIA-PET
def parser_args_hic(parent_parser):
    parser=argparse.ArgumentParser(parents=[parent_parser])
    parser.add_argument('--bins', type=int, default=200)
    parser.add_argument('--crop', type=int, default=4)
    parser.add_argument('--pretrain_path', type=str, default='none')
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--trunk',  type=str, default='transformer')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    args, unknown = parser.parse_known_args()
    return args
### arguments for downstream model to predict 1kb-resolution Micro-C
def parser_args_microc(parent_parser):
    parser=argparse.ArgumentParser(parents=[parent_parser])
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--pretrain_path', type=str, default='none')
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--trunk',  type=str, default='transformer')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    args, unknown = parser.parse_known_args()
    return args

args,parser = get_args()
cage_args=parser_args_cage(parser)
hic_args=parser_args_hic(parser)
microc_args=parser_args_microc(parser)
def load_pre_training_model(saved_model_path,device):
    pretrain_model = build_model(args)
    pretrain_model.to(device)
    pretrain_model.eval()
    pretrain_model.load_state_dict(torch.load(saved_model_path,map_location=device))
    return pretrain_model

def load_cage_model(saved_model_path,device):
    cage_model = build_pretrain_model_cage(cage_args)
    cage_model.to(device)
    cage_model.eval()
    cage_model.load_state_dict(torch.load(saved_model_path))
    return cage_model

def load_hic_model(saved_model_path,device):
    hic_model = build_pretrain_model_hic(hic_args)
    hic_model.to(device)
    hic_model.eval()
    hic_model.load_state_dict(torch.load(saved_model_path))
    return hic_model

def load_microc_model(saved_model_path,device):
    microc_model = build_pretrain_model_microc(microc_args)
    microc_model.to(device)
    microc_model.eval()
    microc_model.load_state_dict(torch.load(saved_model_path))
    return microc_model


def predict_epis(model,chrom, start,end,dnase,fasta_extractor):
    if (end-start)%1000:
        raise ValueError('the length of the input genomic region should be divisible by 1000')
    input_dnase=read_dnase_pickle(dnase,chrom[3:])
    device=next(model.parameters()).device
    inputs = generate_input(fasta_extractor,chrom, start, end, input_dnase).to(device)
    with torch.no_grad():
        pred_epi = torch.sigmoid(model(inputs))
    pred_epi = pred_epi.detach().cpu().numpy()
    return pred_epi

def predict_cage(model,chrom,start,end,dnase,fasta_extractor):
    if (end-start)%200000:
        raise ValueError('the length of the input genomic region should be divisible by 200000')
    input_dnase = read_dnase_pickle(dnase, chrom[3:])
    input_start, input_end = start - 25000, end + 25000
    inputs = []
    for s in range(input_start, input_end - 200000, 200000):
        e = s + 250000
        device = next(model.parameters()).device
        cage_inputs = generate_input(fasta_extractor,chrom, s, e, input_dnase).to(device)
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

def predict_hic(model,chrom,start,end,dnase,fasta_extractor):
    if (end-start)!=1000000:
        raise ValueError('Please input a 1Mb region')
    input_dnase = read_dnase_pickle(dnase, chrom[3:])
    inputs=generate_input(fasta_extractor,chrom,start,end, input_dnase)
    device = next(model.parameters()).device
    inputs=inputs.unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_hic = model(inputs).detach().cpu().numpy().squeeze()
    pred_hic=complete_mat(arraytouptri(pred_hic.squeeze(), hic_args))
    return pred_hic


def predict_microc(model,chrom,start,end,dnase,fasta_extractor):
    if (end-start)!=600000:
        raise ValueError('Please input a 600kb region')
    input_dnase = read_dnase_pickle(dnase, chrom[3:])
    inputs = generate_input(fasta_extractor, chrom, start, end, input_dnase)
    device = next(model.parameters()).device
    inputs = inputs.unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_microc = model(inputs).detach().cpu().numpy().squeeze()
    pred_microc=complete_mat(arraytouptri(pred_microc.squeeze(), microc_args))
    return pred_microc

