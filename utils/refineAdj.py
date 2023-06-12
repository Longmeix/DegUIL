# https://github.com/DSE-MSU/DeepRobust/blob/496573f7cab5b960d5c1f520515eb553b7404bf1/examples/graph/test_prognn.py
import copy
import os
import random
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import time
import numpy as np
import torch
from torch import optim

import config as cfg
from deeprobust.graph.defense.prognn import EstimateAdj
from deeprobust.graph.defense.pgd import PGD, prox_operators

from utils.general import read_pickle, write_pickle
import scipy.sparse as sp

DEVICE = cfg.device


def train_adj(adj, args, net_name):
    adj = ((adj + adj.T) > 0) * 1.
    adj = torch.FloatTensor(adj.todense()).to(DEVICE)
    # estimator = EstimateAdj(adj, symmetric=True, device=DEVICE).to(DEVICE)
    estimator = EstimateAdj(adj, symmetric=False, device=DEVICE).to(DEVICE)
    optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)

    optimizer_l1 = PGD(estimator.parameters(),
                       proxs=[prox_operators.prox_l1],
                       lr=args.lr_adj, alphas=[args.alpha])
    optimizer_nuclear = PGD(estimator.parameters(),
                            proxs=[prox_operators.prox_nuclear_cuda],
                            lr=args.lr_adj, alphas=[args.beta])
    print("\n=== train_adj ===")
    for epoch in range(args.epochs):
        t = time.time()
        estimator.train()
        optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        # normalized_adj = estimator.normalize()
        loss_symmetric = torch.norm(estimator.estimated_adj \
                                    - estimator.estimated_adj.t(), p="fro")

        loss_diffiential = loss_fro + args.phi * loss_symmetric
        loss_diffiential.backward()
        optimizer_adj.step()

        optimizer_nuclear.zero_grad()
        optimizer_nuclear.step()
        loss_nuclear = prox_operators.nuclear_norm

        optimizer_l1.zero_grad()
        optimizer_l1.step()

        total_loss = loss_fro \
                     + args.alpha * loss_l1 \
                     + args.beta * loss_nuclear \
                     + args.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
            estimator.estimated_adj.data, min=0, max=1))

        # if epoch % 1 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_fro: {:4f}'.format(loss_fro.item()),
              'loss_symetric: {:.4f}'.format(loss_symmetric.item()),
              'loss_l1: {:.4f}'.format(loss_l1.item()),
              'loss_nuclear: {:.4f}'.format(loss_nuclear.item()),
              'loss_total: {:.4f}'.format(total_loss.item()))

    # normalized_adj = estimator.normalize()
    # adjRef = to_scipy(normalized_adj.data)
    adjRef = to_scipy(estimator.estimated_adj.data)
    write_pickle(adjRef, os.path.join(folder, f'adjRef_de_{net_name}.pkl'))
    return adjRef


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    indices = tensor.nonzero().t()
    values = tensor[indices[0], indices[1]]
    return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))


# split head vs tail nodes
def split_nodes(adj, low_ratio=0.5, high_ratio=0.1):
    if not isinstance(adj, np.ndarray):
        adj = adj.toarray()
    num_links = np.sum(adj, axis=1)
    head_nodes_num = int(len(num_links) * high_ratio)
    high_degree = np.partition(num_links, kth=-head_nodes_num)[-head_nodes_num]  # select head nodes by the degree
    print('High degree: {}'.format(high_degree))
    idx_high = np.where(num_links > high_degree)[0]
    low_node_num = int(len(num_links) * low_ratio)
    low_degree = np.partition(num_links, kth=low_node_num)[low_node_num]  # select tail and normal nodes by the degree
    if low_degree < 5:
        low_degree = 5
    print('Low degree: {}'.format(low_degree))
    idx_tail = np.where(num_links <= low_degree)[0]

    return idx_high, idx_tail


def sparseAdj(adj, adjRef, net_name=None, low_ratio=0.5, high_ratio=0.1, save_prob=0.7):
    '''
    :param adj: row adjacent, csr_matrix
    :param adjRef: refined adjacent, csr_matrix
    :return:
    '''
    adj = adj.toarray()
    adj = ((adj + adj.T) > 0) * 1.
    high_idx, tail_idx = split_nodes(adj, low_ratio, high_ratio)
    adjRef = adjRef.toarray()
    # delete edges connected to high-degree nodes
    # adjRef_sparse = adj.copy()
    adjRef_sparse = copy.deepcopy(adj)
    adjRef_sparse[high_idx] = adjRef[high_idx]
    adjRef_sparse[:, high_idx] = adjRef[:, high_idx]
    adjRef_sparse[adj == 0] = 0
    # adjRef_sparse[adjRef_sparse < save_prob] = 0
    # adjRef_sparse[adjRef_sparse >= save_prob] = 1

    degrees = np.sum(adj, axis=1)
    save_ratio = 1 - 0.04 / 2
    for i in high_idx:
        n_save_neighbor = int(degrees[i] * save_ratio)
        top_idx = np.argpartition(adjRef_sparse[i], kth=-n_save_neighbor)[-n_save_neighbor:]  # save top similar edges
        adjRef_sparse[i] = 0
        adjRef_sparse[i, top_idx] = 1
    in_degrees = np.sum(adj, axis=0)
    for i in high_idx:
        n_save_neighbor = int(in_degrees[i] * save_ratio)
        top_idx = np.argpartition(adjRef_sparse[:, i], kth=-n_save_neighbor)[-n_save_neighbor:]
        adjRef_sparse[:, i] = 0
        adjRef_sparse[top_idx, i] = 1
    adjRef_sparse[adjRef_sparse < save_prob] = 0
    adjRef_sparse[adjRef_sparse >= save_prob] = 1

    # save original edges connected to tail nodes
    tail_idx = list(set(tail_idx))
    adjRef_sparse[tail_idx] = adj[tail_idx]
    adjRef_sparse[:, tail_idx] = adj[:, tail_idx]

    # make sure no isolated nodes
    single_n = np.where(~adjRef_sparse.any(axis=1))[0]
    adjRef_sparse[single_n] = adj[single_n]
    assert len(np.where(~adj.any(axis=1))[0]) == 0

    adjRef_sparse = ((adjRef_sparse + adjRef_sparse.T) > 1) * 1.
    # adjRef_sparse[((adjRef_sparse + adjRef_sparse.T) < 2)] = 0

    adjRef_sparse = sp.csr_matrix(adjRef_sparse)
    write_pickle(adjRef_sparse, os.path.join(folder, f'adj_ref_{net_name}_1.pkl'))

    n_edges = len(adj.nonzero()[0])
    n_ref_edges = len(adjRef_sparse.nonzero()[0])
    print('Delete net_{} perturb edges: {} (adj: {}, adj_ref: {})'.format(
        net_name, n_edges - n_ref_edges, n_edges, n_ref_edges
    ))

    return adjRef_sparse


def randomDeleteEdges(adj, num_del, net_name=None):
    # adj = (adj + adj.T)
    adjRef_sparse = adj.copy().toarray()
    edges = list(zip(*adj.nonzero()))
    del_edges_idx = random.sample(edges, num_del // 2)
    del_edges_idx = tuple(zip(*del_edges_idx))
    adjRef_sparse[del_edges_idx] = 0
    adjRef_sparse[((adjRef_sparse + adjRef_sparse.T) < 2)] = 0
    adjRef_sparse = sp.csr_matrix(adjRef_sparse)
    write_pickle(adjRef_sparse, os.path.join(folder, f'adj_ref_{net_name}_1.pkl'))

    n_edges = len(adj.nonzero()[0])
    n_ref_edges = len(adjRef_sparse.nonzero()[0])
    print('Delete net_{} perturb edges: {} (adj: {}, adj_ref: {})'.format(
        net_name, n_edges - n_ref_edges, n_edges, n_ref_edges
    ))


def cal_clean_probability(adj, adj_ptb, adj_ref):
    adj = ((adj + adj.T) > 0) * 1.
    ptb_edges = set(zip(*adj_ptb.nonzero()))
    added_ptb_edge = ptb_edges - set(zip(*adj.nonzero()))
    clean_edge = ptb_edges - set(zip(*adj_ref.nonzero()))
    # The proportion of noise edges being cleaned
    cleaned_size = len(added_ptb_edge & clean_edge)
    add_size = len(added_ptb_edge)
    cleaned_prob = cleaned_size / add_size
    print('Cleaned prob: {:.4f}, ({}/{})'.format(cleaned_prob, cleaned_size, add_size))
    return cleaned_prob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FT', help='dataset name, {DBLP, FT}')
    parser.add_argument('--adj_type', type=str, default='row', help='adjacency matrix, {row, perturb}')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='noise ptb_rate')

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
    parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
    parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
    parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('--device', default='cuda:2', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.adj_type == 'perturb':
        folder = os.path.join('../../dataset', args.dataset, 'attack', str(args.ptb_rate))
        adj_s = read_pickle(os.path.join(folder, 'adj_ptb_s.pkl'))
        adj_t = read_pickle(os.path.join(folder, 'adj_ptb_t.pkl'))
    else:
        folder = os.path.join('../../dataset', args.dataset)
        adj_s = read_pickle(os.path.join(folder, 'adj_s.pkl'))
        adj_t = read_pickle(os.path.join(folder, 'adj_t.pkl'))
    low = 0.1

    # estimAdj_s = train_adj(adj_s, args, net_name='s')
    # estimAdj_t = train_adj(adj_t, args, net_name='t')

    adjRef_de_s = read_pickle(os.path.join(folder, 'adjRef_de_s.pkl'))
    adjRef_de_t = read_pickle(os.path.join(folder, 'adjRef_de_t.pkl'))

    # FT
    adj_ref_s = sparseAdj(adj_s, adjRef_de_s, net_name='s', high_ratio=0.1, save_prob=0.5)
    adj_ref_t = sparseAdj(adj_t, adjRef_de_t, net_name='t', high_ratio=0.1, save_prob=0.5)

    # # robust
    # adj_ref_s = sparseAdj(adj_s, adjRef_de_s, net_name='s', low_ratio=0.1, high_ratio=0.1, save_prob=0.5)
    # adj_ref_t = sparseAdj(adj_t, adjRef_de_t, net_name='t', low_ratio=0.1, high_ratio=0.1, save_prob=0.5)

    # randomDeleteEdges(adj_s, num_del=1842, net_name='s')
    # randomDeleteEdges(adj_t, num_del=7322, net_name='t')


    # # DBLP
    # adj_ref_s = sparseAdj(adj_s, adjRef_de_s, net_name='s', high=15, save_prob=0.6)  # epoch 200
    # adj_ref_t = sparseAdj(adj_t, adjRef_de_t, net_name='t', high=15, save_prob=0.6)

    # for net in ('s', 't'):
    #     adj = read_pickle(os.path.join('../../dataset', args.dataset, f'adj_{net}.pkl'))
    #     adj_ref = read_pickle(f'../../dataset/{args.dataset}/attack/{args.ptb_rate}/adj_ref_{net}_1.pkl')
    #     adj_ptb = read_pickle(f'../../dataset/{args.dataset}/attack/{args.ptb_rate}/adj_ptb_{net}.pkl')
    #     print(f'---- {args.dataset} ptb_rate:{args.ptb_rate} {net}_net ----')
    #     cleaned_prob = cal_clean_probability(adj, adj_ptb, adj_ref)





