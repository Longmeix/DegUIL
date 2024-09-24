import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append(rootPath)

import argparse
import torch
import config as cfg
import random
import numpy as np
from utils.general import read_pickle, write_pickle
import networkx as nx
from node2vec import Graph
from gensim.models import Word2Vec
from MLPmap import MLPmap


def seed_torch(seed=2022):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(data_dir, save_emb_path, ratio):
    adj_s = read_pickle(data_dir + 'adj_s.pkl').astype(np.float32)
    adj_t = read_pickle(data_dir + 'adj_t.pkl').astype(np.float32)
    embeds = read_pickle(save_emb_path)
    links = read_pickle(data_dir + f'links_{ratio}.pkl')

    return adj_s, adj_t, embeds, links


def get_emb(adj, emb_dim, p=1, q=1, directed=False,
            num_walks=10, walk_length=80,
            window_size=5, workers=8, epochs=5):
    if dataset == 'D_W_15K_V1':
        num_walks = 100
        walk_length = 8
    adj = ((adj + adj.T) > 0) * 1.
    print('Creating graph...')
    g = nx.from_numpy_matrix(adj, create_using=nx.Graph())
    h = Graph(g, directed, p, q)
    print('Creating alias table...')
    h.preprocess_transition_probs()
    print('Random walking...')
    walks = h.simulate_walks(num_walks, walk_length)
    walks = [list(map(str, walk)) for walk in walks]
    print('Training...')
    model = Word2Vec(walks, size=emb_dim, window=window_size,
                     min_count=0, negative=5, sg=1, cbow_mean=1,
                     workers=workers, iter=epochs)
    emb = np.array(list(map(model.wv.get_vector, map(str, range(g.number_of_nodes())))))
    # write_pickle(embs, save_emb_path)
    return emb


def train_embeddings(adj_path, emb_dim, save_emb_path):
    embs = []
    for suffix in ('s', 't'):
        adj = read_pickle(adj_path.format(suffix))
        adj = ((adj + adj.T) > 0) * 1.
        embs.append(get_emb(adj, emb_dim))
    write_pickle(embs, save_emb_path)


def train_align(adj_s, adj_t, emb, links, args):
    # ins = MANA(emb, adj_s, adj_t, all_links, args)
    # ins.train_mapping(args.mapping_epochs)
    adj_t = ((adj_t + adj_t.T) > 0) * 1.
    ins = MLPmap(emb, links, adj_t, args)
    ins.train()


def parse_args():
  parser = argparse.ArgumentParser(description='Node2Vec')
  parser.add_argument('--folder_dir', default='./datasets/', type=str)
  parser.add_argument('--dataset', default='FT', help='FT, DBLP, D_W_15K_V1', type=str)
  parser.add_argument('--device', default='cuda:3', type=str)
  # embeds
  parser.add_argument('--model', default='Node2Vec', type=str)
  parser.add_argument('--embed_dim', type=int, default=cfg.dim_feature, help='Embedding dimension')
  parser.add_argument('--mapping_lr', default=0.005, type=float)
  parser.add_argument('--mapping_epochs', default=70, type=int)
  parser.add_argument('--ratio', default=0.5, type=float)
  parser.add_argument('--top_k', default=cfg.k, type=int)

  args = parser.parse_args()
  # print(args)
  return args


if __name__ == '__main__':
    os.chdir('../')
    seed_torch(2023)
    args = parse_args()
    cfg.init_args(args)
    cfg.epochs = args.mapping_epochs
    # file path
    data_dir = args.folder_dir + args.dataset + '/'
    adj_path = data_dir + 'adj_{}.pkl'
    save_emb_path = data_dir + 'emb_n2v1.pkl'
    # save_emb_path = data_dir + 'emb_deepwalk.pkl'

    # train embeddings
    is_train = False
    if not os.path.exists(save_emb_path) or is_train:
        dataset = args.dataset
        train_embeddings(adj_path, cfg.dim_feature, save_emb_path)
    # train mapping function
    train_align(*load_data(data_dir, save_emb_path, args.ratio), args)