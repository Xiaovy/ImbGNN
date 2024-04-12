import numpy as np
import torch
from torch_geometric.utils.dropout import dropout_edge
import time
import copy
import random

def dropout_edge(edge_index, p=0.5, edge=None):
    if p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index
    # deal edge
    edge_mask = torch.rand(row.size(0), device=edge_index.device) < p
    if edge is not None:
        edge_mask = torch.logical_and(edge_mask, edge)
    
    edge_index = edge_index[:, ~edge_mask]

    return edge_index, edge_mask


def remove_edge(edge_index, drop_ratio, edge=None):
    edge_index, _ = dropout_edge(edge_index, p = drop_ratio, edge=edge)
    return edge_index


def drop_node(x, drop_ratio):
    node_num, _ = x.size()
    drop_num = int(node_num * drop_ratio)

    idx_mask = np.random.choice(node_num, drop_num, replace = False).tolist()

    x[idx_mask] = 0

    return x
