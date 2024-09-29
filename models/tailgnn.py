from torch.nn.parameter import Parameter
import math
from layers.graphconv import GraphConv
from layers.gat import *


class Relation(nn.Module):
    def __init__(self, in_features, ablation):
        super(Relation, self).__init__()

        self.gamma_1 = nn.Linear(in_features, in_features, bias=False)
        self.gamma_2 = nn.Linear(in_features, in_features, bias=False)

        self.beta_1 = nn.Linear(in_features, in_features, bias=False)
        self.beta_2 = nn.Linear(in_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()
        self.ablation = ablation

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, ft, neighbor):

        if self.ablation == 3:
            self.m = ft + self.r - neighbor
        else:
            gamma = self.gamma_1(ft) + self.gamma_2(neighbor)
            gamma = self.lrelu(gamma) + 1.0

            beta = self.beta_1(ft) + self.beta_2(neighbor)
            beta = self.lrelu(beta)

            self.r_v = gamma * self.r + beta

            # transE
            self.m = ft + self.r_v - neighbor
            '''
            #transH
            norm = F.normalize(self.r_v) 
            h_ft = ft - norm * torch.sum((norm * ft), dim=1, keepdim=True)
            h_neighbor = neighbor - norm * torch.sum((norm * neighbor), dim=1, keepdim=True)
            self.m = h_ft - h_neighbor
            '''
        return self.m  # F.normalize(self.m)


class TransGCN(nn.Module):
    def __init__(self, nfeat, nhid, device, ablation=0):
        super(TransGCN, self).__init__()

        self.device = device
        self.ablation = ablation

        self.r = Relation(nfeat, ablation)
        self.gc = GraphConv(nfeat, nhid)

    def forward(self, x, adj, head):
        mean = F.normalize(adj, p=1, dim=1)
        neighbor = torch.mm(mean, x)
        output = self.r(x, neighbor)
        adj = adj + torch.eye(adj.size(0), device=self.device)

        if head or self.ablation == 2:
            # norm = F.normalize(adj, p=1, dim=1)
            h_k = self.gc(x, mean)
        else:
            h_s = torch.mm(output, self.gc.weight)  # missing neighbor information
            h_k = self.gc(x, adj)  # observed neighborhood
            h_k = h_k + h_s  # ideal neighborhood

            num_neighbor = torch.sum(adj, dim=1, keepdim=True)
            h_k = h_k / (num_neighbor + 1)

        return h_k, output  # h_k: ideal, output: miss. info


class TransGAT(nn.Module):
    def __init__(self, nfeat, nhid, device, ablation=0, nheads=3, dropout=0., concat=True):
        super(TransGAT, self).__init__()

        self.device = device
        self.ablation = ablation
        self.r = Relation(nfeat, ablation)

        self.gat = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=0.2, concat=concat) for _ in
                    range(nheads)]
        for i, attention in enumerate(self.gat):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj, head):
        mean = F.normalize(adj, p=1, dim=1)
        neighbor = torch.mm(mean, x)

        output = self.r(x, neighbor)
        adj = adj + torch.eye(adj.size(0), device=self.device)
        edge = adj.nonzero(as_tuple=False).t()

        if head or self.ablation == 2:
            h_k = torch.cat([att(x, edge) for att in self.gat], dim=1)
        else:
            h_k = torch.cat([att(x, edge, mi=output) for att in self.gat], dim=1)

        return h_k, output


# latent relation GCN
class TailGNN(nn.Module):
    def __init__(self, nfeat, out_dim, params, device):
        super(TailGNN, self).__init__()
        self.nhid = params.hidden
        self.dropout = params.dropout
        self.gnn_type = params.gnn_type

        if self.gnn_type == 1:
            self.rel1 = TransGCN(nfeat, self.nhid, device=device,ablation=params.ablation)
            self.rel2 = TransGCN(nfeat + self.nhid, out_dim, device=device, ablation=params.ablation)

        elif self.gnn_type == 2:
            nheads = 3
            nhid = 8
            self.rel1 = TransGAT(nfeat, nhid, device=device, ablation=params.ablation,
                                 nheads=nheads, dropout=self.dropout, concat=True)
            self.rel2 = TransGAT(nhid * nheads, out_dim, device=device, ablation=params.ablation,
                                 nheads=1, dropout=self.dropout, concat=False)
        else:
            self.rel1_gcn = TransGCN(nfeat, self.nhid, device=device, ablation=params.ablation)
            self.rel2_gcn = TransGCN(nfeat + self.nhid, out_dim, device=device, ablation=params.ablation)
            self.rel1_gat = TransGAT(nfeat, self.nhid, device=device, ablation=params.ablation,
                                     nheads=1, concat=False)
            self.rel2_gat = TransGAT(nfeat + self.nhid, out_dim, device=device, ablation=params.ablation,
                                     nheads=1, concat=False)

    def forward(self, x, adj, head):
        if self.gnn_type != 3:
            x1, out1 = self.rel1(x, adj, head)
            # x1 = F.elu(x1)
            x1 = torch.cat([x, x1], dim=1)
            # x1 = F.dropout(x1, self.dropout, training=self.training)
            x2, out2 = self.rel2(x1, adj, head)
            return x2, [out1, out2]
        else:
            x_gcn, out1_gcn = self.rel1_gcn(x, adj, head)
            x_gat, out1_gat = self.rel1_gat(x, adj, head)
            x_gcn = torch.cat([x, x_gcn], dim=1)
            x_gcn, out2_gcn = self.rel2_gcn(x_gcn, adj, head)

            x_gat = torch.cat([x, x_gat], dim=1)
            x_gat, out2_gat = self.rel2_gat(x_gat, adj, head)

            return torch.cat([x_gcn, x_gat], dim=1), [out1_gcn, out1_gat, out2_gcn, out2_gat]