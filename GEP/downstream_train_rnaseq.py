from rnaseq.model import build_pre_train_model
import os, pickle, time
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn import metrics
import argparse
from rnaseq_dataset import GexDataset,TestDataset
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=11)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--shuffle_dataset', default=True, action='store_false', help='model testing')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--test', default=True, action='store_false', help='model testing')
    parser.add_argument('--ac_data', type=str, default='dnase_norm')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    parser.add_argument('--finetune_model', type=str, default='lstm',choices=['cnn','lstm'])
    parser.add_argument('--pretrain_path', type=str, default='none')
    # parser.add_argument('--pretrain_path',type=str,default='/nfs/turbo/umms-drjieliu/usr/zzh/KGbert/pretrain/models/checkpoint_pretrain_cnn1_dnase_norm.pt')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()
    model= build_pre_train_model(args)
    model.cuda()

    criterion= torch.nn.BCEWithLogitsLoss()

    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)


    dataset=GexDataset()
    dataset_size=len(dataset)
    indices=np.arange(dataset_size)
    valid_split=int(np.floor(dataset_size*0.85))
    print(dataset_size,valid_split)

    cls = ['HepG2', 'HMEC', 'HSMM', 'K562']

    if args.test:
        testdatasets={}
        test_loaders={}
        for cl in cls:
            testdatasets[cl] = TestDataset(cl)
            test_loaders[cl]=DataLoader(testdatasets[cl], batch_size=args.batchsize)
    if args.shuffle_dataset:
        print('shuffle datatset')
        np.random.seed(8)
        np.random.shuffle(indices)
    train_indices,valid_indices=indices[:valid_split],indices[valid_split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(dataset, batch_size=args.batchsize,
                                               sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=args.batchsize,
                                                    sampler=valid_sampler)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best_loss=1e10
    for epoch in range(args.epochs):
        training_losses=[]
        model.train()
        for step, (input_seq,input_label) in enumerate(train_loader):
            t = time.time()
            train_seq= input_seq.float().to(device)
            train_target = input_label.float().to(device)
            output = model(train_seq)
            loss = criterion(output, train_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_loss = loss.item()
            training_losses.append(cur_loss)
            if step % 1000== 0:
                print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "train_loss=",
                              "{:.7f}".format(cur_loss),
                              "time=", "{:.5f}".format(time.time() - t), "lr=", "{:.10f}".format(optimizer.param_groups[0]['lr'])
                              )
        validation_losses = []
        pred_eval = []
        target_eval = []
        model.eval()
        for step, (input_seq,input_label) in enumerate(valid_loader):
            t = time.time()
            valid_seq = input_seq.float().to(device)
            valid_target = input_label.float().to(device)
            with torch.no_grad():
                output = model(valid_seq)
                loss = criterion(output, valid_target)
                cur_loss = loss.item()
                validation_losses.append(cur_loss)
                pred_eval.append(torch.sigmoid(output).cpu().data.detach().numpy())
                target_eval.append(valid_target.cpu().data.detach().numpy())
        pred_eval = np.concatenate(pred_eval, axis=0)
        target_eval = np.concatenate(target_eval, axis=0)
        auc = metrics.roc_auc_score(target_eval, pred_eval)
        train_loss = np.average(training_losses)
        print('Epoch: {} LR: {:.8f} train_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], train_loss))
        valid_loss = np.average(validation_losses)
        print('Epoch: {} LR: {:.8f} valid_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], valid_loss))
        if valid_loss < best_loss:
            print('save model')
            torch.save(model.state_dict(),
                       'models/gex_classification_%s_%s.pt'%(args.finetune_model,args.ac_data))
            best_loss = valid_loss
            if args.test:
                for cl in cls:
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

                    testing_losses = []
                    pred_eval = []
                    target_eval = []
                    model.eval()
                    with torch.no_grad():
                        for step, (input_seq,input_label) in enumerate(test_loaders[cl]):
                            t = time.time()
                            test_seq = input_seq.float().to(device)

                            test_target = input_label.float().to(device)
                            output = model(test_seq)
                            loss = criterion(output, test_target)
                            cur_loss = loss.item()
                            testing_losses.append(cur_loss)
                            pred_eval.append(torch.sigmoid(output).cpu().data.detach().numpy())
                            target_eval.append(test_target.cpu().data.detach().numpy())
                    pred_eval = np.concatenate(pred_eval, axis=0)
                    target_eval = np.concatenate(target_eval, axis=0)
                    auc = metrics.roc_auc_score(target_eval, pred_eval)
                    ap=metrics.average_precision_score(target_eval, pred_eval)
                    test_loss = np.average(testing_losses)
                    print(auc, ap)

if __name__=='__main__':
    main()
