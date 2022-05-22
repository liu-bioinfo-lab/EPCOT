from torch.utils.data import Dataset
import pickle, torch
import numpy as np
class Task1Dataset(Dataset):
    def __init__(self,ac_type):
        cls=['K562', 'MCF-7', 'GM12878', 'HepG2']
        pretrain_path='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/pretrain_data/'
        with open(pretrain_path+'train_seqs.pickle','rb') as f:
            temp_seq=pickle.load(f)
        with open(pretrain_path+'train_%s.pickle'%ac_type,'rb') as f:
            temp_dnase=pickle.load(f)
        with open(pretrain_path+'train_labels.pickle','rb') as f:
            temp_labels=pickle.load(f)
        self.input_seq=torch.tensor(np.vstack([temp_seq[cl] for cl in cls]))
        self.input_dnase=torch.tensor(np.vstack([temp_dnase[cl] for cl in cls]))
        self.input_label=torch.tensor(np.vstack([temp_labels[cl].toarray() for cl in cls]))
        temp_label_mask=[]
        for i in range(len(cls)):
            temp_label_mask+= [i]* (temp_labels[cls[i]].shape[0])
        self.label_mask=torch.tensor(np.array(temp_label_mask))
        self.num=self.input_seq.shape[0]
    def __getitem__(self, index):
        return self.input_seq[index],self.input_dnase[index],self.input_label[index],self.label_mask[index]
    def __len__(self):
        return self.num