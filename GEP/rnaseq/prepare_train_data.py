### prepare training data

import pickle,os,csv
import numpy as np
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

def get_gene_tss():
    trans_start = {}
    with open('TSS.txt', 'r') as f:
        f.readline()
        for line in f:
            contents = line.strip().split(',')
            geneid = contents[0]
            chr = contents[4]
            tss = int(contents[7])
            strand = int(contents[9])
            try:
                temp = trans_start[geneid].copy()
                if strand == -1:
                    if tss > temp[1]:
                        trans_start[geneid] = [chr, tss]
                else:
                    if tss < temp[1]:
                        trans_start[geneid] = [chr, tss]
            except Exception:
                trans_start[geneid] = [chr, tss]
    with open('gene_tss.pickle', 'wb') as f:
        pickle.dump(trans_start, f)
    return trans_start

def generate_input_dnase(cl,genes,gene_tss):
    dnase_data = {}
    with open('/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/normalize_dnase/%s_dnase.pickle'%cl,'rb') as f:
        dnase=pickle.load(f)
    temp_dnase = {}
    tempchrs = [i for i in np.arange(1, 23)] + ['X']
    for chr in tempchrs:
        temp_dnase[str(chr)] = np.expand_dims(pad_signal_matrix(dnase[chr].toarray().reshape(-1, 1000)), axis=1)
        print(cl, chr, temp_dnase[str(chr)].shape)
    for gene in genes:
        chr = gene_tss[gene][0]
        tss_idx= gene_tss[gene][1]//1000
        indices=np.arange(tss_idx-5,tss_idx+6)
        dnase_data[gene]=temp_dnase[str(chr)][indices]
    with open('data/%s_train_dnase.pickle'%cl,'wb') as f:
        pickle.dump(dnase_data,f)




def generate_input_seq(genes,gene_tss,ref_genomes):
    input_seqs={}
    for gene in genes:
        chr=gene_tss[gene][0]
        tss_idx= gene_tss[gene][1]//1000
        indices=np.arange(tss_idx-5,tss_idx+6)
        refs=ref_genomes[chr][indices]
        if np.sum(refs)== refs.shape[0]*refs.shape[-1]:
            input_seqs[gene]=refs
    return input_seqs

def label_gene(genes,file):
    genelabels={}
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            geneid='ENSG00000'+ row[0]
            label=int(row[-1])
            if geneid in genes:
                genelabels[geneid]=label
    return genelabels


def main():
    ref_path = '/scratch/drjieliu_root/drjieliu/zhenhaoz/ref_genome'
    chr_list = [i for i in np.arange(1, 23)] + ['X']
    ref_genomes = {}
    for chr in chr_list:
        ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
        ref_gen_data = load_npz(ref_file).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
        ref_gen_data = pad_seq_matrix(ref_gen_data)
        ref_genomes[str(chr)] = ref_gen_data
        print(chr, ref_gen_data.shape)

    gene_tss = get_gene_tss()
    ## gene set aligned to hg38
    train_genes = np.load('train_genes1.npy')
    training_seqs=generate_input_seq(train_genes,gene_tss,ref_genomes)
    with open('data/train_seq_data.pickle','wb') as f:
        pickle.dump(training_seqs,f)

    eid = ['E003', 'E114', 'E116', 'E117', 'E118', 'E119', 'E120', 'E123']
    cls = ['H1', 'A549', 'GM12878', 'HeLa-S3', 'HepG2', 'HMEC', 'HSMM', 'K562']
    for cl in cls:
        generate_input_dnase(cl, train_genes, gene_tss)

    for i in range(len(eid)):
        id=eid[i]
        cl=cls[i]
        print(id,cl)
        train_labels=label_gene(train_genes,'/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/gene_exp/data/label_data/%s/classification/train.csv'%id)
        with open('data/%s_train_labels.pickle' % cl, 'wb') as f:
            pickle.dump(train_labels,f)




if __name__=="__main__":
    main()




