import pickle,csv
import numpy as np
from torch.utils.data import Dataset
import torch

data_path='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/gene_exp/data/'
class GexDataset(Dataset):
    def __init__(self):
        eid = ['E003', 'E114', 'E116', 'E117', 'E118', 'E119', 'E120', 'E123']
        cls = ['H1', 'A549', 'GM12878', 'HeLa-S3']
        with open(data_path+'train_seq_data.pickle','rb') as f:
            train_seq_data=pickle.load(f)
        train_genes=train_seq_data.keys()

        with open(data_path+'valid_seq_data.pickle','rb') as f:
            valid_seq_data=pickle.load(f)
        valid_genes=valid_seq_data.keys()

        labels=[]
        traindata=[]
        for cl in cls:
            with open(data_path+'%s_train_dnase.pickle'%cl,'rb') as f:
                dnase_data=pickle.load(f)
            with open(data_path+'%s_valid_dnase.pickle' % cl, 'rb') as f:
                vdnase_data = pickle.load(f)
            with open(data_path+'%s_train_labels.pickle'%cl,'rb') as f:
                label_data=pickle.load(f)
            with open(data_path+'%s_valid_labels.pickle'%cl,'rb') as f:
                vlabel_data=pickle.load(f)
            for gene in train_genes:
                traindata.append(np.concatenate((train_seq_data[gene],dnase_data[gene]),axis=1))
                labels.append(label_data[gene])
            for gene in valid_genes:
                traindata.append(np.concatenate((valid_seq_data[gene],vdnase_data[gene]),axis=1))
                labels.append(vlabel_data[gene])
        self.seqs=torch.tensor(np.stack(traindata))
        self.labels=torch.tensor(labels).reshape(-1,1)
        self.num=self.seqs.shape[0]
        print(self.seqs.shape,self.labels.shape)
    def __getitem__(self, index):
        return self.seqs[index],self.labels[index]
    def __len__(self):
        return self.num

class TestDataset(Dataset):
    def __init__(self,cl):
        labels = []
        testdata = []
        with open(data_path+'test_seq_data.pickle','rb') as f:
            test_seq_data = pickle.load(f)
        test_genes = test_seq_data.keys()
        with open(data_path+'%s_test_dnase.pickle' % cl, 'rb') as f:
            dnase_data = pickle.load(f)

        with open(data_path+'%s_test_labels.pickle' % cl, 'rb') as f:
            label_data = pickle.load(f)
        for gene in test_genes:
            testdata.append(np.concatenate((test_seq_data[gene], dnase_data[gene]), axis=1))
            labels.append(label_data[gene])
        self.seqs = torch.tensor(np.stack(testdata))
        self.labels = torch.tensor(labels).reshape(-1,1)
        self.num = self.seqs.shape[0]

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]

    def __len__(self):
        return self.num


