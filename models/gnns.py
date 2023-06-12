import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from layers import *


class TransGCN(nn.Module):
    def __init__(self, nfeat, nhid, device):
        super(TransGCN, self).__init__()
        self.device = device
        self.gc = GraphConv(nfeat, nhid)

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=self.device)
        norm = F.normalize(adj, p=1, dim=1)
        h_k = self.gc(x, norm)

        return h_k  # h_k: ideal, output: miss. info


class TransGAT(nn.Module):
    def __init__(self, nfeat, nhid, device, nheads=3, dropout=0.5, concat=True):
        super(TransGAT, self).__init__()
        self.device = device
        self.gat = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=0.2, concat=concat) for _ in
                    range(nheads)]
        for i, attention in enumerate(self.gat):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=self.device)
        edge = adj.nonzero(as_tuple=False).t()
        h_k = torch.cat([att(x, edge) for att in self.gat], dim=1)

        return h_k


class TailGNN(nn.Module):
    def __init__(self, nfeat, nclass, params, device):
        super(TailGNN, self).__init__()
        self.nhid = params.hidden
        self.dropout = params.dropout
        self.act = nn.Tanh()

        self.rel1_gcn = TransGCN(nfeat, self.nhid, device=device)
        self.rel2_gcn = TransGCN(self.nhid, self.nhid,  device=device)
        nheads = 3
        nhid = 8
        self.rel1_gat = TransGAT(nfeat, nhid, device=device, nheads=nheads, dropout=self.dropout,
                                 concat=True)
        self.rel2_gat = TransGAT(nhid * nheads, self.nhid, device=device, nheads=1, dropout=self.dropout, concat=False)

        self.w_gcn = nn.Parameter(self.init_randn_uni(self.nhid, nclass))
        self.w_gat = nn.Parameter(self.init_randn_uni(self.nhid, nclass))
        self.act = nn.Tanh()

    def forward(self, x, adj):
        alpha = 0.1
        x1_gcn = self.rel1_gcn(x, adj)
        x2_gcn = self.rel2_gcn(x1_gcn, adj)

        x1_gat = self.rel1_gat(x, adj)
        x2_gat = self.rel2_gat(x1_gat, adj)
        x2 = x2_gcn @ self.w_gcn + alpha * x2_gat @ self.w_gat
        x2 = self.act(x2)

        return torch.cat([x, x2], dim=1)

    @staticmethod
    def init_randn_uni(size, dim):
        emb = nn.Parameter(torch.randn(size, dim))
        emb.data = F.normalize(emb.data)
        return emb