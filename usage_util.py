import argparse
import kipoiseq
from kipoiseq import Interval
import numpy as np
import torch,pickle
from pretraining.model import build_model
from pretraining.track.track_model import build_track_model
from GEP.cage.model import build_pretrain_model_cage
from COP.hic.model import build_pretrain_model_hic
from COP.microc.model import build_pretrain_model_microc


def parser_args():
    """
    Hyperparameters for the pre-training model
    """
    # add_help = False
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('--num_class', default=245, type=int,help='the number of epigenomic features to be predicted')
    parser.add_argument('--seq_length', default=1600, type=int,help='the length of input sequences')
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--load_backbone', default=False)
    args, unknown = parser.parse_known_args()
    return args,parser
def get_args():
    args,_ = parser_args()
    return args,_

def parser_args_epi_track(parent_parser):
    """
    Hyperparameters for the downstream model to predict 1kb-resolution CAGE-seq
    """
    parser=argparse.ArgumentParser(parents=[parent_parser],add_help = False)
    parser.add_argument('--bins', type=int, default=500)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--pretrain_path', type=str, default='none')
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--fine_tune', default=False, action='store_true')
    parser.add_argument('--return_embed', default=False, action='store_true')
    args, unknown = parser.parse_known_args()
    return args


def parser_args_cage(parent_parser):
    """
    Hyperparameters for the downstream model to predict 1kb-resolution CAGE-seq
    """
    parser=argparse.ArgumentParser(parents=[parent_parser],add_help = False)
    parser.add_argument('--bins', type=int, default=250)
    parser.add_argument('--crop', type=int, default=25)
    parser.add_argument('--pretrain_path', type=str, default='none')
    parser.add_argument('--embed_dim', default=360, type=int)
    parser.add_argument('--mode', type=str, default='transformer')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    args, unknown = parser.parse_known_args()
    return args

def parser_args_hic(parent_parser):
    """
    Hyperparameters for the downstream model to predict 5kb-resolution Hi-C and ChIA-PET
    """
    parser=argparse.ArgumentParser(parents=[parent_parser],add_help = False)
    parser.add_argument('--bins', type=int, default=200)
    parser.add_argument('--crop', type=int, default=4)
    parser.add_argument('--pretrain_path', type=str, default='none')
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--trunk',  type=str, default='transformer')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    args, unknown = parser.parse_known_args()
    return args

def parser_args_microc(parent_parser):
    """
    Hyperparameters for the downstream model to predict 1kb-resolution Micro-C
    """
    parser=argparse.ArgumentParser(parents=[parent_parser],add_help = False)
    parser.add_argument('--bins', type=int, default=600)
    parser.add_argument('--crop', type=int, default=50)
    parser.add_argument('--pretrain_path', type=str, default='none')
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--trunk',  type=str, default='transformer')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    args, unknown = parser.parse_known_args()
    return args

args,parser = get_args()
epi_track_args=parser_args_epi_track(parser)
cage_args=parser_args_cage(parser)
hic_args=parser_args_hic(parser)
microc_args=parser_args_microc(parser)


def load_input_dnase(
        pickle_file
):
    """
    Args:
        pickle_file (str): the path to a pickle file storing DNase-seq
    Returns:
        dict: a python dictionary storing DNase-seq
            {chromosome (int): a sparse matrix representing DNase-seq signals (scipy.sparse.csr.csr_matrix)}
    """
    with open(pickle_file,'rb') as f:
        dnase=pickle.load(f)
    return dnase

def read_dnase_pickle(
        dnase_in_pickle,
        chrom
):
    """
    Args:
        dnase_in_pickle (dict): DNase-seq data in dictionary structure
        chrom (int or str): chromosome
    Returns:
        ndarray: a numpy array representing DNase-seq signals in the chromsome of interest
    """
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
    """
    search an TF in the list of predicted epigenomic features
    Args:
        tf: the name of an epigenomic feature
    Returns:
        int: the index of the epigenomic feature in the list
    """
    with open('EPCOT/Profiles/epigenomes.txt', 'r') as f:
        epigenomes = f.read().splitlines()
    try:
        tf_idx= epigenomes.index(tf)
    except Exception:
        raise ValueError("%s is not in the list of predicted TFs"%tf)
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


def generate_input(
        fasta_extractor,
        chrom, start, end,
        dnase
):
    """
    Generate the inputs to the model
    Args:
        fasta_extractor (kipoiseq.extractors.FastaStringExtractor): kipoiseq.extractors.FastaStringExtractor object
        chrom (str), start (int), end (int): Specify a genomic region (chromosome, start genomic position, and end genomic position),
                and the size of the genomic region should be divisible by 1000.
        dnase (ndarray): a numpy array representing DNase-seq signals in the same chromosome

    Returns:
        torch.Tensor: a torch tensor (N x 5 x 1600) representing input genomic sequences and DNase-seq, where N represents the number of
                1kb sequences in the input genomic region
    """

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

def load_pre_training_model(
        saved_model_path,
        device
):
    """
    Load the pre-training model.
    Args:
        saved_model_path (str): the path to the saved pre-training model.
        device: an object representing the device ('cpu' or 'cuda') where the model will be allocated.

    Examples:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        saved_model='models/pretrain_dnase.pt'
        pretrain_model=load_pre_training_model(saved_model,device)
    """
    pretrain_model = build_model(args)
    model_dict = pretrain_model.state_dict()
    pretrain_dict = torch.load(saved_model_path, map_location='cpu')
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    pretrain_model.load_state_dict(model_dict)
    pretrain_model.eval()
    pretrain_model.to(device)
    return pretrain_model

def load_epi_track_model(
        saved_model_path,
        device
):
    track_model = build_track_model(epi_track_args)
    model_dict = track_model.state_dict()
    pretrain_dict = torch.load(saved_model_path, map_location='cpu')
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    track_model.load_state_dict(model_dict)
    track_model.eval()
    track_model.to(device)
    return track_model

def load_cage_model(
        saved_model_path,
        device
):
    """
    Load EPCOT model to predict 1kb-resolution CAGE-seq.
    """
    cage_model = build_pretrain_model_cage(cage_args)
    model_dict = cage_model.state_dict()
    downstream_dict = torch.load(saved_model_path, map_location='cpu')
    downstream_dict = {k: v for k, v in downstream_dict.items() if k in model_dict}
    model_dict.update(downstream_dict)
    cage_model.load_state_dict(model_dict)
    cage_model.eval()
    cage_model.to(device)
    return cage_model

def load_hic_model(
        saved_model_path,
        device
):
    """
    Load EPCOT model to predict 5kb-resolution Hi-C or ChIA-pet contact maps.
    """
    hic_model = build_pretrain_model_hic(hic_args)
    model_dict = hic_model.state_dict()
    downstream_dict = torch.load(saved_model_path, map_location='cpu')
    downstream_dict = {k: v for k, v in downstream_dict.items() if k in model_dict}
    model_dict.update(downstream_dict)
    hic_model.load_state_dict(model_dict)
    hic_model.eval()
    hic_model.to(device)
    return hic_model

def load_microc_model(
        saved_model_path,
        device
):
    """
    Load EPCOT model to predict 1kb-resolution Micro-C contact maps.
    """
    microc_model = build_pretrain_model_microc(microc_args)
    model_dict = microc_model.state_dict()
    downstream_dict = torch.load(saved_model_path, map_location='cpu')
    downstream_dict = {k: v for k, v in downstream_dict.items() if k in model_dict}
    model_dict.update(downstream_dict)
    microc_model.load_state_dict(model_dict)
    microc_model.eval()
    microc_model.to(device)
    return microc_model


def predict_epis(
        model,
        chrom, start, end,
        dnase,
        fasta_extractor
):
    """
    Predict epigenomic features on 1kb sequences.
    Args:
        model: the pre-training model
        chrom (str), start (int), end (int): Specify a genomic region (chromosome, start genomic position, and end genomic position),
                and the size of the genomic region should be divisible by 1000.
        dnase (dict): a python dictionary storing DNase-seq
        fasta_extractor: kipoiseq.extractors.FastaStringExtractor object

    Returns:
        ndarray: a numpy matrix (N x 245) of predicted binding activities of 245 epigenomic features, where N represents the number of
            1kb sequences in the input genomic region

    Examples:
        pretrain_model = load_pre_training_model(saved_model,device)
        fasta_extractor = FastaStringExtractor(fasta_file)
        chrom,start,end = ['chr11',46750000,47750000]
        GM12878_dnase=load_input_dnase(input_dnase_file)

        pred_score_epi = predict_epis(model=pretrain_model,
                            chrom=chrom, start=start,end=end,
                            dnase=GM12878_dnase,
                            fasta_extractor=fasta_extractor)
    """
    if (end-start)%1000:
        raise ValueError('the length of the input genomic region should be divisible by 1000')
    input_dnase=read_dnase_pickle(dnase,chrom[3:])
    device=next(model.parameters())
    inputs = generate_input(fasta_extractor,chrom, start, end, input_dnase).to(device)
    with torch.no_grad():
        pred_epi = torch.sigmoid(model(inputs))
    pred_epi = pred_epi.detach().cpu().numpy()
    return pred_epi

def predict_track(
        model,
        chrom,start,end,
        dnase,
        fasta_extractor,
        cross_cell_type=False
    ):
    if (end-start)!=400000:
        raise ValueError('Please input a 400kb region')
    input_dnase = read_dnase_pickle(dnase, chrom[3:])
    input_start, input_end = start - 50000, end + 50000
    inputs=generate_input(fasta_extractor,chrom,input_start, input_end, input_dnase)
    device = next(model.parameters()).device
    inputs=inputs.unsqueeze(0).float().to(device)
    if cross_cell_type:
        for m in model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                m.train()
    with torch.no_grad():
        pred_hic = model(inputs).detach().cpu().numpy().squeeze()
    pred_hic=complete_mat(arraytouptri(pred_hic.squeeze(), hic_args))
    return pred_hic


def predict_cage(
        model,
        chrom,start,end,
        dnase,
        fasta_extractor,
        cross_cell_type=False
    ):
    """
    1kb CAGE-seq prediction
    Args:
        model: EPCOT model to predict CAGE-seq
        chrom (str), start (int), end (int): Specify a genomic region (chromosome, start genomic position, and end genomic position),
            and the size of the genomic region should be divisible by 200000
        dnase (dict): a python dictionary storing DNase-seq
        fasta_extractor (kipoiseq.extractors.FastaStringExtractor): kipoiseq.extractors.FastaStringExtractor object
        cross_cell_type (bool): if performing cross-cell type prediction

    Returns:
        ndarray: a numpy array of predicted CAGE-seq. Each element indicates the predicted CAGE-seq signal on a 1kb region
    """
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

    if cross_cell_type:
        pred_cage = []
        for m in model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                m.train()
        with torch.no_grad():
            for i in range(inputs.shape[0]):
                pred_cage.append(model(inputs[i:i+1]).detach().cpu().numpy().squeeze())
        pred_cage = np.concatenate(pred_cage)
    else:
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


def predict_hic(model,chrom,start,end,dnase,fasta_extractor,cross_cell_type=False):
    if (end-start)!=1000000:
        raise ValueError('Please input a 1Mb region')
    input_dnase = read_dnase_pickle(dnase, chrom[3:])
    inputs=generate_input(fasta_extractor,chrom,start,end, input_dnase)
    device = next(model.parameters()).device
    inputs=inputs.unsqueeze(0).float().to(device)
    if cross_cell_type:
        for m in model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                m.train()
    with torch.no_grad():
        pred_hic = model(inputs).detach().cpu().numpy().squeeze()
    pred_hic=complete_mat(arraytouptri(pred_hic.squeeze(), hic_args))
    return pred_hic


def predict_microc(model,chrom,start,end,dnase,fasta_extractor,cross_cell_type=False):
    if (end-start)!=600000:
        raise ValueError('Please input a 600kb region')
    input_dnase = read_dnase_pickle(dnase, chrom[3:])
    inputs = generate_input(fasta_extractor, chrom, start, end, input_dnase)
    device = next(model.parameters()).device
    inputs = inputs.unsqueeze(0).float().to(device)
    if cross_cell_type:
        for m in model.modules():
            if m.__class__.__name__.startswith('BatchNorm'):
                m.train()
    with torch.no_grad():
        pred_microc = model(inputs).detach().cpu().numpy().squeeze()
    pred_microc=complete_mat(arraytouptri(pred_microc.squeeze(), microc_args))
    return pred_microc

import pyfaidx
class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

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
def plot_hic(ax, mat,cmap='RdBu_r', vmin=0, vmax=5):
    ax.imshow(mat,cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])