import numpy as np
from scipy.sparse import csr_matrix,save_npz

one_hot_dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
def stringtoonehot(string):
    one_hot=np.zeros((4,len(string)))
    for i in range(len(string)):
        if string[i].upper() != 'N':
            one_hot[one_hot_dic[string[i].upper()],i]=1
    return one_hot

def fatomatrix(fa_file):
    with open(fa_file,'r') as f:
        temp=f.readlines()
    ref_genome=[]
    for line in temp:
        if line.startswith('>'):
            continue
        else:
            seq_string=line.strip()
            seq_onehot=stringtoonehot(seq_string)
            ref_genome.append(seq_onehot)
    ref_genome=np.hstack(ref_genome).astype('int8')
    seq_length =ref_genome.shape[1]//1000*1000
    print(seq_length)
    return csr_matrix(ref_genome[:,:seq_length])

chroms=[str(i) for i in range(23)]+['X']
for chr in chroms:
    # change the path to fasta file
    fa_file = 'chr%s.fa' % chr
    # change the saved location
    save_npz('chr%s.npz' % (chr), fatomatrix(fa_file))

