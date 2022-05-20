import numpy as np
import torch,os,pickle
from sklearn import metrics
import multiprocessing as mp

def cal_metrics(v):
    true, pred = v
    idx = np.where(true > 0)
    if idx[0].tolist():
        auc=metrics.roc_auc_score(true,pred)
        ap=metrics.average_precision_score(true,pred)

    else:
        auc=np.nan
        ap=np.nan
    return auc,ap
def best_param(preds,targets,cl,preserve=False):
    # preds_eval = np.concatenate(preds,axis=0)
    # targets_eval = np.concatenate(targets,axis=0)
    pool = mp.Pool(32)
    result = pool.map_async(cal_metrics, ((targets[:, i], preds[:, i]) for i in range(preds.shape[1])))
    pool.close()
    pool.join()
    r = np.array(result.get())
    r=r[~np.isnan(r)].reshape(-1,2)
    if preserve:
        np.save('metrics/%s.npy'%cl,r)
    return np.mean(r[:,0]),np.mean(r[:,1])

