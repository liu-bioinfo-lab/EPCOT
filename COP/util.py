import straw
import numpy as np
import os
from scipy.sparse import load_npz
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

def load_refgenome():
    ref_genome={}
    ref_path='/scratch/drjieliu_root/drjieliu/zhenhaoz/ref_genome/'
    chrs=[str(i) for i in range(1,23)]+['X']
    for chr in chrs:
        ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
        ref_gen_data = load_npz(ref_file).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
        print(chr, ref_gen_data.shape)
        ref_genome[chr]=pad_seq_matrix(ref_gen_data)
    return ref_genome

def txttomatrix(txt_file,resolution):
    rows=[]
    cols=[]
    data=[]
    with open(txt_file,'r') as f:
        for line in f:
            contents=line.strip().split('\t')
            bin1=int(contents[0])//resolution
            bin2 = int(contents[1]) // resolution
            if np.abs(bin2-bin1)>500:
                continue
            value=float(contents[2])
            rows.append(bin1)
            cols.append(bin2)
            data.append(value)
    return np.array(rows),np.array(cols),np.array(data)


