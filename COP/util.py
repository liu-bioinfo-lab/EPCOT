import pickle,os
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
    ref_path = '/scratch/drjieliu_root/drjieliu/zhenhaoz/ref_genome/mm10/'
    # ref_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/data/ref_genome/'
    ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
    ref_gen_data = load_npz(ref_file).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
    return torch.tensor(pad_seq_matrix(ref_gen_data))

def load_dnase(dnase_seq):
    dnase_seq = np.expand_dims(pad_signal_matrix(dnase_seq.reshape(-1, 1000)), axis=1)
    return torch.tensor(dnase_seq)

def load_hic(cl,chr):
    hic_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/3D/OE_matrix/'
    hic = load_npz(hic_path + '%s/chr%s_5kb.npz' % (cl, chr))
    return hic

def prepare_train_data(cl,chrs):
    dnase_data={}
    ref_data={}
    hic_data={}
    dnase_path = '/scratch/drjieliu_root/drjieliu/zhenhaoz/DNase/normalize_dnase/'
    # dnase_path = '/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/normalize_dnase/'
    with open(dnase_path + '%s_dnase.pickle' % cl, 'rb') as f:
        dnase = pickle.load(f)
    for chr in chrs:
        if chr != 'X':
            dnase_seq=dnase[int(chr)]
        else:
            dnase_seq = dnase['X'].reshape(1,-1)

        dnase_data[chr]=load_dnase(dnase_seq.toarray())
        ref_data[chr]=load_ref_genome(chr)
        hic_data[chr]=load_hic(cl,chr)
    return dnase_data, ref_data,hic_dat

