from torch.utils.data import Dataset as BaseDataset
from torch_geometric.data.collate import collate
import torch
from utils import *
from torch_geometric.utils import subgraph, degree
from aug import *
from torch_sparse import SparseTensor, matmul
import sys
import traceback
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
import time


class Dataset_imb(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset

    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]

        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])
        train_idx = torch.arange(batch_id.shape[0])
        pad_knn_id = find_knn_id(batch_id, self.args.kernel_idx)
        feed_dicts.extend([self.all_dataset[i] for i in pad_knn_id])

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )

        knn_edge_index, _ = subgraph(
            data.id, self.args.knn_edge_index, relabel_nodes=True)

        knn_edge_index, _ = add_remaining_self_loops(knn_edge_index)
        row, col = knn_edge_index
        knn_deg = degree(col, data.id.shape[0])
        deg_inv_sqrt = knn_deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]

        knn_adj_t = torch.sparse.FloatTensor(
            knn_edge_index, edge_weight, (data.id.size(0), data.id.size(0)))
        
        
        aug_xs, aug_adj_ts = [], []

        node_map = data.id[data.batch]
        aug_node = self.args.aug_node[node_map]
        aug_edge = self.args.aug_edge[node_map]
        row, col = data.adj_t.coo()[:2]
        edge_mask = aug_edge[row] & aug_edge[col]

        for i in range(self.args.aug_num):
            edge_index = torch.stack(data.adj_t.coo()[:2])
            edge_index_aug = remove_edge(edge_index, self.args.drop_edge_ratio, edge_mask)
            
            aug_adj_ts.append(SparseTensor(
                row=edge_index_aug[0], col=edge_index_aug[1], value=None, sparse_sizes=(data.x.size(0), data.x.size(0))))
            
            tmpx = drop_node(data.x, self.args.mask_node_ratio)
            tmpx[~aug_node] = data.x[~aug_node]
            aug_xs.append(tmpx)

        batch = {'data': data,
                 'train_idx': train_idx,
                 'aug_adj_ts': aug_adj_ts,
                 'aug_xs': aug_xs,
                 'knn_adj_t': knn_adj_t}
        return batch
