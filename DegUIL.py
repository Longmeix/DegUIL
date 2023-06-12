import os
from itertools import chain
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
from utils import data_process
from models.deggnn import DegGNN
from models.UILAggregator import UILAggregator, MLP
import config as cfg

# Get parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='FT', help='dataset: FT, DBLP')
parser.add_argument("--model", type=str, default='DegUIL', help='model name: DegUIL, Tail-GNN')
parser.add_argument('--ratio', default=0.5, type=float)
parser.add_argument("--gnn_type", type=int, default=3, help='1: gcn, 2: gat, 3: gcn+gat')
parser.add_argument("--hidden", type=int, default=64, help='hidden layer dimension')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--mu", type=float, default=0.001, help='missing/redundant info constraint')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--D", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--seed", type=int, default=2022, help='Random seed')
parser.add_argument("--epochs", type=int, default=1200, help='Epochs')
parser.add_argument("--device", type=str, default='cuda:1', help='gpu id or cpu')

args = parser.parse_args()
cfg.init_args(args)
dataset = args.dataset
model = args.model
DEVICE = args.device
# DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'dataset:{dataset}, model:{args.model}')


def seed_torch(seed=2022):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))
    sum_m /= len(out_feat)
    return sum_m


def mse_loss(emb_h, emb_t, idx):
    loss = torch.mean(torch.norm(emb_h[idx] - emb_t[idx]))
    return loss


def train_embed():
    embed_super_s, mis_s_s, _ = embed_model(features[0], adj_s, deg='super')
    embed_super_t, mis_s_t, _ = embed_model(features[1], adj_t, deg='super')
    embed_norm_s, mis_n_s, rdd_n_s = embed_model(features[0], adj_s, deg='norm')
    embed_norm_t, mis_n_t, rdd_n_t = embed_model(features[1], adj_t, deg='norm')
    embed_tail_s, _, rdd_t_s = embed_model(features[0], tail_adj_s, deg='tail')
    embed_tail_t, _, rdd_t_t = embed_model(features[1], tail_adj_t, deg='tail')
    # loss
    L_uil_super = uil_h.train(embed_super_s, embed_super_t, common)
    L_uil_h = uil_h.train(embed_norm_s, embed_norm_t, common)
    L_uil_t = uil_h.train(embed_tail_s, embed_tail_t, common)
    L_uil = L_uil_super + L_uil_h + L_uil_t

    m_sup_s = normalize_output(mis_s_s, idx_super_s)
    m_sup_t = normalize_output(mis_s_t, idx_super_t)
    m_sup = m_sup_s + m_sup_t

    m_norm_s = normalize_output(mis_n_s + rdd_n_s, idx_norm_s)  # Loss for potential information constraint
    m_norm_t = normalize_output(mis_n_t + rdd_n_t, idx_norm_t)
    m_norm = m_norm_s + m_norm_t

    m_tail_s = normalize_output(rdd_t_s, idx_tail_s)
    m_tail_t = normalize_output(rdd_t_t, idx_tail_t)
    m_tail = m_tail_s + m_tail_t
    m = (m_sup + m_norm + m_tail) / 3

    L_all = L_uil + args.mu * m

    optimizer.zero_grad()
    L_all.backward()
    optimizer.step()

    return L_all

seed_torch(args.seed)
features, (adj_s, adj_t), links, idx_s, idx_t = data_process.load_dataset(dataset, model, args.ratio, folder='./datasets/', k=args.D)
features = [emb.to(cfg.device) for emb in features]

# forge tail node
tail_adj_s = data_process.link_dropout(adj_s, idx_s[0])
extend_adj_s = data_process.extend_edges(adj_s, idx_s[0])
adj_s = torch.FloatTensor(adj_s.todense()).to(DEVICE)
extend_adj_s = torch.FloatTensor(extend_adj_s.todense()).to(DEVICE)
tail_adj_s = torch.FloatTensor(tail_adj_s.todense()).to(DEVICE)
tail_adj_t = data_process.link_dropout(adj_t, idx_t[0])
extend_adj_t = data_process.extend_edges(adj_t, idx_t[0])
adj_t = torch.FloatTensor(adj_t.todense()).to(DEVICE)
extend_adj_t = torch.FloatTensor(extend_adj_t.todense()).to(DEVICE)
tail_adj_t = torch.FloatTensor(tail_adj_t.todense()).to(DEVICE)

# indexes of train nodes and test nodes
idx_norm_s = torch.LongTensor(idx_s[0])
idx_super_s = torch.LongTensor(idx_s[1])
idx_tail_s = torch.LongTensor(idx_s[2])
idx_norm_t = torch.LongTensor(idx_t[0])
idx_super_t = torch.LongTensor(idx_t[1])
idx_tail_t = torch.LongTensor(idx_t[2])

print("Data Processing done!")

out_dim = cfg.dim_feature

# Model and optimizer
embed_model = DegGNN(nfeat=features[0].shape[1], out_dim=out_dim, params=args, device=DEVICE).to(DEVICE)

dim = out_dim * 2

common = nn.ModuleList([
            MLP(dim, cfg.dim_feature),
            MLP(dim, cfg.dim_feature)
        ]).to(DEVICE)

optimizer = optim.Adam(
            chain(embed_model.parameters(),
                  common.parameters(),
                  ),
            lr=args.lr,
            # weight_decay=args.lamda
            )

# Train model
t_total = time.time()
uil_h = UILAggregator(adj_s, adj_t, links, k=cfg.k)

print_info_epochs = 10
loss_epc = .0
for epoch in range(args.epochs):
    t = time.time()

    embed_model.train()
    common.train()

    loss_batch = train_embed()
    loss_epc += loss_batch

    # Evaluation:
    embed_model.eval()
    common.eval()

    embed_super_s, _, _ = embed_model(features[0], adj_s, deg='super')
    embed_super_t, _, _ = embed_model(features[1], adj_t, deg='super')
    embed_tail_s, _, _ = embed_model(features[0], adj_s, deg='tail')
    embed_tail_t, _, _ = embed_model(features[1], adj_t, deg='tail')
    embed_tail_s[idx_super_s] = embed_super_s[idx_super_s]
    embed_tail_t[idx_super_t] = embed_super_t[idx_super_t]
    uil_h.eval_hit(epoch, embed_tail_s, embed_tail_t, common)

    if epoch % print_info_epochs == 0:
        log = 'Epoch: {:d} '.format(epoch + 1) + \
              'loss_train: {:.4f} '.format(loss_batch.item())
        print(log)

uil_h.print_performance_k()

print("Training Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))