import os

from numpy import test
from parse import parse_args
from tqdm import tqdm
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import math

from utils import *
from model import *
from learn import *
from dataset import *
from dataprocess import *


def run(args):
    pbar = tqdm(range(args.runs), unit='run')

    F1_micro = np.zeros(args.runs, dtype=float)
    F1_macro = np.zeros(args.runs, dtype=float)

    Dataset_use = Dataset_imb

    for count in pbar:
        random.seed(args.seed + count)
        np.random.seed(args.seed + count)
        torch.manual_seed(args.seed + count)
        torch.cuda.manual_seed_all(args.seed + count)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rand_node = torch.rand(args.degb.shape)
        rand_edge = torch.rand(args.degb.shape)
        args.aug_edge = rand_edge <= args.p_edge
        args.aug_node = rand_node <= args.p_node

        train_data, val_data, test_data = shuffle(
            dataset, args.c_train_num, args.c_val_num, args.y)
        
        train_data = upsample(train_data)
        val_data = upsample(val_data)

        train_dataset = Dataset_use(train_data, dataset, args)
        val_dataset = Dataset_use(val_data, dataset, args)
        test_dataset = Dataset_use(test_data, dataset, args)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_dataset.collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=val_dataset.collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=test_dataset.collate_batch)

        encoder = GIN(args, use_drop=args.use_drop).to(args.device)
        classifier = MLP_Classifier(args).to(args.device)

        optimizer_e = torch.optim.Adam(
            encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_c = torch.optim.Adam(
            classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_loss = math.inf
        val_loss_hist = []

        for epoch in range(0, args.epochs):
            
            loss = train(encoder, classifier, train_loader,
                         optimizer_e, optimizer_c, args)
            val_eval = eval(encoder, classifier, val_loader, args)

            if(val_eval['loss'] < best_val_loss):
                best_val_loss = val_eval['loss']

                test_eval = eval(encoder, classifier, test_loader, args)

            val_loss_hist.append(val_eval['loss'])

            if(args.early_stopping > 0 and epoch > args.epochs // 2):
                tmp = torch.tensor(
                    val_loss_hist[-(args.early_stopping + 1): -1])
                if(val_eval['loss'] > tmp.mean().item()):
                    break

        F1_micro[count] = test_eval['F1-micro']
        F1_macro[count] = test_eval['F1-macro']

    return F1_micro, F1_macro


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()
    
    dataset, args.n_feat, args.n_class, _ = get_TUDataset(
        args.dataset, pre_transform=T.ToSparseTensor())

    args.c_train_num, args.c_val_num = get_class_num(
        args.imb_ratio, args.num_train, args.num_val)

    args.y = torch.tensor([data.y.item() for data in dataset])
    size_lst = [data.x.shape[0] for data in dataset]
    args.size = torch.tensor(size_lst)
    size_lst.sort()
    args.mid = size_lst[int(args.size_ratio * len(size_lst))]
    args.sizeb = args.size >= args.mid

    args.degs = []
    for data in dataset:
        edge_index = torch.stack(data.adj_t.coo()[:2])
        row, col = edge_index
        deg = degree(col)
        args.degs.append(deg.mean().item())
    degs = sorted(args.degs)
    args.middeg = degs[int(args.deg_ratio * len(degs))]
    args.degs = torch.tensor(args.degs)
    args.degb = args.degs > args.middeg
    scal = np.log(degs[-1] / degs[0])
    args.p_edge = (args.aug_ratio - (1 - args.aug_ratio)) * torch.log(args.degs / degs[0]) / scal + 1 - args.aug_ratio
    args.p_node = ((1 - args.aug_ratio) - args.aug_ratio) * torch.log(args.degs / degs[0]) / scal + args.aug_ratio

    args.kernel_idx, args.knn_edge_index = get_kernel_knn(args.dataset, args.kernel_type, args.knn_nei_num, args.sizeb)

    F1_micro, F1_macro = run(args)

    print('F1_macro: ', np.mean(F1_macro))
    print('F1_micro: ', np.mean(F1_micro))
