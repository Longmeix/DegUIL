import pickle
import os
import numpy as np
import torch
from scipy import sparse as sp

folder = 'datasets/'


def link_dropout(adj, idx, k=5):
    tail_adj = adj.copy()
    num_links = np.random.randint(k, size=idx.shape[0])
    num_links += 1

    probs = np.abs(adj.toarray() > 0).sum(axis=1) ** (3 / 4)

    for i in range(idx.shape[0]):
        index = tail_adj[idx[i]].nonzero()[1]
        prob = probs[index]
        prob = prob / prob.sum()
        new_idx = np.random.choice(index, num_links[i], replace=False, p=prob)
        tail_adj[idx[i]] = 0.0
        for j in new_idx:
            tail_adj[idx[i], j] = 1.0
    return tail_adj


def extend_edges(adj, idx, add_prob=0.1):
    if not isinstance(adj, np.ndarray):
        adj = adj.toarray()
    extend_adj = adj.copy()
    degree = np.sum(adj, axis=1)
    head_nodes_num = int(len(degree) * 0.1)
    high_degree = np.partition(degree, kth=-head_nodes_num)[-head_nodes_num]
    add_num_links = (high_degree - degree + 1 + np.random.randint(5, size=adj.shape[0])).astype(np.int32)

    probs = np.abs(adj > 0).sum(axis=1) ** (-3 / 4)

    for i in range(idx.shape[0]):
        index = np.where(extend_adj[idx[i]] == 0)[0]
        prob = probs[index]
        prob = prob / prob.sum()
        new_idx = np.random.choice(index, add_num_links[idx[i]], replace=False, p=prob)
        for j in new_idx:
            extend_adj[idx[i], j] = 1.0
    extend_adj = sp.lil_matrix(extend_adj)
    return extend_adj


def split_nodes_by_ratio(adj, low=5, high_ratio=0.1):
    if not isinstance(adj, np.ndarray):
        adj = adj.toarray()
    num_links = np.sum(adj, axis=1)
    head_nodes_num = int(len(num_links) * high_ratio)
    high_degree = np.partition(num_links, kth=-head_nodes_num)[-head_nodes_num]  # select head nodes by the degree
    idx_super = np.where(num_links >= high_degree)[0]
    idx_tail = np.where(num_links <= low)[0]
    idx_norm_tail = np.where(num_links < high_degree)[0]
    idx_norm = np.setdiff1d(idx_norm_tail, idx_tail)
    print(high_degree)
    return idx_norm, idx_super, idx_tail


def read_pickle(infile):
    with open(infile, 'rb') as f:
        return pickle.load(f)


def symmetric_adj(adjPath):
    adj = read_pickle(adjPath)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tolil()
    # remove self connection
    for i in range(adj.shape[0]):
        adj[i, i] = 0.

    return adj


def standardization(data):
    mu = torch.mean(data, dim=0)
    sigma = torch.std(data, dim=0)
    return (data - mu) / sigma


def process_data(model, path, ratio='degree', k=5):
    assert model in ['Tail-GNN', 'DegUIL']
    adj_s = symmetric_adj(path + "adj_s.pkl")
    adj_t = symmetric_adj(path + "adj_t.pkl")
    features = read_pickle(path + "emb_n2v1.pkl")

    if isinstance(features[0], torch.Tensor):
        features = [emb.detach() for emb in features]
    else:
        features = [torch.FloatTensor(emb) for emb in features]
    features = [standardization(emb) for emb in features]

    # label head tail for train/test
    idx_train_s, idx_super_s, idx_tail_s = split_nodes_by_ratio(adj_s, low=5, high_ratio=0.1)
    idx_train_t, idx_super_t, idx_tail_t = split_nodes_by_ratio(adj_t, low=5, high_ratio=0.1)

    # link label
    if ratio == 0:
        links = read_pickle(path + "links_degree.pkl")
    else:
        links = read_pickle(path + f"links_{ratio}.pkl")

    return features, (adj_s, adj_t), links, (idx_train_s, idx_super_s, idx_tail_s), \
           (idx_train_t, idx_super_t, idx_tail_t)


def load_dataset(dataset, model, ratio=0.5, folder=None, k=10):
    path = folder + dataset + '/'

    DATASET = {
        'FT': process_data,
        'DBLP': process_data,
    }

    if dataset not in DATASET:
        return ValueError('Dataset not available')
    else:
        print('Preprocessing data ...')
        return DATASET[dataset](model, path, ratio, k=k)


if __name__ == "__main__":
    _, adj, _, _ = load_dataset('FT', 'DegUIL', ratio='', path='./datasets/', k=5)
