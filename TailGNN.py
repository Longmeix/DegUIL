from itertools import chain
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from models.UILAggregator import UILAggregator, MLP
import os, argparse
from utils import *
from layers import Discriminator
from models.tailgnn import TailGNN
import config as cfg

print(os.getcwd())
# Get parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='FT', help='dataset')
parser.add_argument("--model", type=str, default='Tail-GNN', help='model name: Tail-GNN')
parser.add_argument('--ratio', default=1., type=float)
parser.add_argument("--hidden", type=int, default=64, help='hidden layer dimension')
parser.add_argument("--gnn_type", type=int, default=1, help='1: gcn, 2: gat, 3: gcn+gat')
parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
parser.add_argument("--eta", type=float, default=0.1, help='adversarial constraint')
parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--D", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--seed", type=int, default=2022, help='Random seed')
parser.add_argument("--epochs", type=int, default=700, help='Epochs')
# parser.add_argument("--patience", type=int, default=300, help='Patience')
parser.add_argument("--id", type=int, default=2, help='gpu ids')
# parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')
parser.add_argument("--device", type=str, default='cuda:0', help='gpu id or cpu')


# DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
cfg.init_args(args)
dataset = args.dataset
model = args.model
# is_robust = args.robust_experiment
DEVICE = cfg.device
criterion = nn.BCELoss()

def seed_torch(seed=2022):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)


def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))
    return sum_m


def train_disc(disc, feature, embed_model, batch, adj, tail_adj, h_labels, t_labels, optimizer_D):
    disc.train()
    optimizer_D.zero_grad()

    embed_h, _ = embed_model(feature, adj, True)
    embed_t, _ = embed_model(feature, tail_adj, False)

    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    # loss
    errorD = criterion(prob_h[batch], h_labels)
    errorG = criterion(prob_t[batch], t_labels)
    L_d = (errorD + errorG) / 2

    L_d.backward()
    optimizer_D.step()
    return L_d


def train_embed(epoch):
    embed_h_s, support_h_s = embed_model[0](features[0], adj_s, True)
    embed_h_t, support_h_t = embed_model[1](features[1], adj_t, True)
    embed_t_s, _ = embed_model[0](features[0], tail_adj_s, False)
    embed_t_t, _ = embed_model[1](features[1], tail_adj_t, False)
    # loss
    L_uil_h = uil_h.train(embed_h_s, embed_h_t, common)
    L_uil_t = uil_h.train(embed_t_s, embed_t_t, common)
    L_uil = (L_uil_h + L_uil_t) / 2  # uil task loss
    # weight regularizer
    m_h_s = normalize_output(support_h_s, idx_train_s)  # Loss for missing information constraint
    m_h_t = normalize_output(support_h_t, idx_train_t)
    m_h = (m_h_s + m_h_t) / 2

    prob_t_s = disc_s(embed_t_s)
    prob_t_t = disc_t(embed_t_t)
    L_d_s = criterion(prob_t_s[idx_train_s], t_labels_s)
    L_d_t = criterion(prob_t_t[idx_train_t], t_labels_t)
    L_d = (L_d_s + L_d_t) / 2

    L_all = L_uil + args.mu * m_h - (args.eta * L_d)

    optimizer.zero_grad()
    L_all.backward()
    optimizer.step()

    return L_all


features, (adj_s, adj_t), links, idx_s, idx_t = data_process.load_dataset(dataset, args.model, args.ratio, folder='./datasets/', k=args.D)
if isinstance(features[0], torch.Tensor):
    features = [emb.to(cfg.device) for emb in features]
else:
    features = [torch.FloatTensor(emb).to(cfg.device) for emb in features]


# forge tail node
tail_adj_s = data_process.link_dropout(adj_s, idx_s[0], k=args.D)
adj_s = torch.FloatTensor(adj_s.todense()).to(DEVICE)
tail_adj_s = torch.FloatTensor(tail_adj_s.todense()).to(DEVICE)
tail_adj_t = data_process.link_dropout(adj_t, idx_t[0], k=args.D)
adj_t = torch.FloatTensor(adj_t.todense()).to(DEVICE)
tail_adj_t = torch.FloatTensor(tail_adj_t.todense()).to(DEVICE)

# indexes of train nodes and test nodes
idx_train_s = torch.LongTensor(idx_s[0])
idx_test_s = torch.LongTensor(idx_s[1])
idx_train_t = torch.LongTensor(idx_t[0])
idx_test_t = torch.LongTensor(idx_t[1])

print("Data Processing done!")

# r_ver = 1
out_dim = cfg.dim_feature

# Model and optimizer
embed_model = nn.ModuleList(
                [TailGNN(nfeat=features[0].shape[1], out_dim=out_dim, params=args, device=DEVICE),
                 TailGNN(nfeat=features[1].shape[1], out_dim=out_dim, params=args, device=DEVICE)]).to(DEVICE)

dim = out_dim
common = nn.ModuleList([
            MLP(dim, cfg.dim_feature),
            MLP(dim, cfg.dim_feature)
        ]).to(DEVICE)

optimizer = optim.Adam(
            chain(embed_model.parameters(),
                  common.parameters()
                  ),
            lr=args.lr,
            weight_decay=args.lamda
            )


feat_disc = dim
disc_s = Discriminator(feat_disc).to(DEVICE)
optimizer_D_s = optim.Adam(disc_s.parameters(),
                         lr=args.lr, weight_decay=args.lamda)
disc_t = Discriminator(feat_disc).to(DEVICE)
optimizer_D_t = optim.Adam(disc_t.parameters(),
                         lr=args.lr, weight_decay=args.lamda)


# head node' label 1, tail node' label 0
h_labels_s = torch.full((len(idx_train_s), 1), 1.0, device=DEVICE)
t_labels_s = torch.full((len(idx_train_s), 1), 0.0, device=DEVICE)
h_labels_t = torch.full((len(idx_train_t), 1), 1.0, device=DEVICE)
t_labels_t = torch.full((len(idx_train_t), 1), 0.0, device=DEVICE)

# Train model
t_total = time.time()
uil_h = UILAggregator(adj_s, adj_t, links, k=cfg.k)

steps_per_epoch = 10
for epoch in range(args.epochs):
    t = time.time()

    loss_epc = .0

    L_d_s = train_disc(disc_s, features[0], embed_model[0], idx_train_s, adj_s, tail_adj_s,
                       h_labels_s, t_labels_s, optimizer_D_s)
    L_d_t = train_disc(disc_t, features[1], embed_model[1], idx_train_t, adj_t, tail_adj_t,
                        h_labels_t, t_labels_t, optimizer_D_t)

    embed_model.train()
    common.train()

    loss_batch = train_embed(epoch)
    loss_epc += loss_batch

    # validate:
    embed_model.eval()
    common.eval()
    embed_h_s, _ = embed_model[0](features[0], adj_s, False)
    embed_h_t, _ = embed_model[1](features[1], adj_t, False)

    uil_h.eval_hit(epoch, embed_h_s, embed_h_t, common)

    if epoch % steps_per_epoch == 0:
        log = 'Epoch: {:d} '.format(epoch + 1) + \
              'loss_train: {:.4f} '.format(loss_batch.item())
        print(log)

uil_h.print_performance_k()

print("Training Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Testing
# print('Test ...')
# embed_model = torch.load(os.path.join(save_path, 'model.pt'))
# test()
