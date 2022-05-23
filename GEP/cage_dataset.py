import numpy as np
from torch.utils.data import Dataset
import torch,pickle

def log_norm(x):
    return np.log2(x+1)
class inputDataset(Dataset):
    def __init__(self,input_locs,labels,cls_idx,crop=25):
        indices=[]
        for idx in cls_idx:
            pad_cl=idx*np.ones((input_locs.shape[0],1))
            indices.append(np.hstack((input_locs, pad_cl)))
        self.indices = np.vstack(indices)
        self.labels= torch.tensor(log_norm(labels[:,crop:-crop])).unsqueeze(-1)
        self.num = self.indices.shape[0]
        print(self.indices.shape,self.labels.shape,self.num)
    def __getitem__(self, index):
        return self.indices[index],self.labels[index]
    def __len__(self):
        return self.num

class clDataset(Dataset):
    def __init__(self,input_locs,labels,cidx,crop=25):
        pad_cl=cidx*np.ones((input_locs.shape[0],1))
        self.indices = np.hstack((input_locs, pad_cl))
        self.labels=torch.tensor(log_norm(labels[:,crop:-crop])).unsqueeze(-1)
        self.num = self.indices.shape[0]
        print(self.indices.shape,self.labels.shape,self.num)
    def __getitem__(self, index):
        return self.indices[index],self.labels[index]
    def __len__(self):
        return self.num
