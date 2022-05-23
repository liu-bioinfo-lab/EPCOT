import pickle,os,h5py
import numpy as np
from scipy.sparse import load_npz,csr_matrix,save_npz
import torch
def pad_seq_matrix(matrix, pad_len=300):
    # add flanking region to each sample
    paddings = np.zeros((1, 4, pad_len)).astype('int8')
    dmatrix = np.concatenate((paddings, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], paddings), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)

def pad_signal_matrix(matrix, pad_len=300):
    paddings = np.zeros(pad_len).astype('float32')
    dmatrix = np.vstack((paddings, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], paddings))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))

def load_ref_genome(chr):
    ref_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/data/ref_genome/'
    ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
    ref_gen_data = load_npz(ref_file).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
    return torch.tensor(pad_seq_matrix(ref_gen_data))

def load_dnase(dnase_seq):
    dnase_seq = np.expand_dims(pad_signal_matrix(dnase_seq.reshape(-1, 1000)), axis=1)
    return torch.tensor(dnase_seq)
def load_cage(cl):
    cage_file='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/gene_exp/ct_cage/data/%s_seq_cov.h5'%cl
    with h5py.File(cage_file, 'r') as hf:
        cage_data = np.array(hf['targets'])
    return cage_data

def prepare_train_data(cls):
    dnase_data={}
    ref_data={}
    cage_data={}
    chroms=[str(i) for i in range(1,23)]
    for chr in chroms:
        ref_data[chr] = load_ref_genome(chr)
    for cl in cls:
        cage_data[cl]=load_cage(cl)
        dnase_data[cl]={}
        dnase_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/normalize_dnase/'
        with open(dnase_path + '%s_dnase.pickle' % cl, 'rb') as f:
            dnase = pickle.load(f)
        for chr in range(1,23):
            dnase_data[cl][str(chr)]=load_dnase(csr_matrix(dnase[chr]).toarray())
    return dnase_data, ref_data,cage_data





