from layers.degree_balance import *
from layers.graphconv import GraphConv
from layers.gat import *


class TransGCN(nn.Module):
    def __init__(self, nfeat, nhid, device):
        super(TransGCN, self).__init__()

        self.device = device

        self.balance = BalanceFeature(nfeat)
        self.r = Relation(nfeat)
        self.gc = GraphConv(nfeat, nhid)

    def forward(self, x, adj, deg):
        mean = F.normalize(adj, p=1, dim=1)
        neighbor = torch.mm(mean, x)
        # missing = self.missing(x, neighbor)
        # redundancy = self.redundancy(x, neighbor)
        missing, redundancy = self.balance(x, neighbor)
        adj = adj + torch.eye(adj.size(0), device=self.device)

        if deg == 'norm':
            h_k = self.gc(x, mean)
        else:
            h_k = self.gc(x, adj)  # observed neighborhood
            if deg == 'super':
                h_s = torch.mm(missing, self.gc.weight)  # missing neighbor information
                h_k = h_k - h_s  # ideal neighborhood
            elif deg == 'tail':
                # redundancy = self.redundancy(x, neighbor)
                h_s = torch.mm(redundancy, self.gc.weight)  # missing neighbor information
                h_k = h_k + h_s  # ideal neighborhood

            num_neighbor = torch.sum(adj, dim=1, keepdim=True)
            h_k = h_k / (num_neighbor + 1)

        return h_k, missing, redundancy  # h_k: ideal, output: miss. info


class TransGAT(nn.Module):
    def __init__(self, nfeat, nhid, device, nheads=3, dropout=0., concat=True):
        super(TransGAT, self).__init__()

        self.device = device
        self.balance = BalanceFeature(nfeat)
        self.r = Relation(nfeat)

        self.gat = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=0.2, concat=concat) for _ in
                    range(nheads)]
        for i, attention in enumerate(self.gat):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj, deg='norm'):
        mean = F.normalize(adj, p=1, dim=1)
        neighbor = torch.mm(mean, x)
        # missing = self.missing(x, neighbor)
        # redundancy = self.redundancy(x, neighbor)
        missing, redundancy = self.balance(x, neighbor)

        adj = adj + torch.eye(adj.size(0), device=self.device)
        edge = adj.nonzero(as_tuple=False).t()

        if deg == 'norm':
            h_k = torch.cat([att(x, edge) for att in self.gat], dim=1)
        else:
            if deg == 'super':
                h_k = torch.cat([att(x, edge, mi=-redundancy) for att in self.gat], dim=1)
            elif deg == 'tail':
                h_k = torch.cat([att(x, edge, mi=missing) for att in self.gat], dim=1)

        return h_k, missing, redundancy


# latent relation GCN
class DegGNN(nn.Module):
    def __init__(self, nfeat, out_dim, params, device):
        super(DegGNN, self).__init__()
        self.nhid = params.hidden
        self.dropout = params.dropout
        self.gnn_type = params.gnn_type

        if self.gnn_type == 1:
            self.rel1 = TransGCN(nfeat, self.nhid, device=device)
            self.rel2 = TransGCN(nfeat + self.nhid, out_dim, device=device)

        elif self.gnn_type == 2:
            nheads = 3
            nhid = 8
            self.rel1 = TransGAT(nfeat, nhid, device=device,
                                 nheads=nheads, dropout=self.dropout, concat=True)
            self.rel2 = TransGAT(nhid * nheads, out_dim, device=device,
                                 nheads=1, dropout=self.dropout, concat=False)
        else:
            self.rel1_gcn = TransGCN(nfeat, self.nhid, device=device)
            self.rel2_gcn = TransGCN(nfeat + self.nhid, out_dim, device=device)
            self.rel1_gat = TransGAT(nfeat, self.nhid, device=device,
                                     nheads=1, concat=False)
            self.rel2_gat = TransGAT(nfeat + self.nhid, out_dim, device=device,
                                     nheads=1, concat=False)

    def forward(self, x, adj, deg):
        if self.gnn_type != 3:
            x1, mis1, redun1 = self.rel1(x, adj, deg)
            x1 = torch.cat([x, x1], dim=1)
            x2, mis2, redun2 = self.rel2(x1, adj, deg)
            return x2, [mis1, redun1, mis2, redun2]
        else:
            x_gcn, mis_gcn1, r_gcn_1 = self.rel1_gcn(x, adj, deg)
            x_gat, mis_gat1, r_gat_1 = self.rel1_gat(x, adj, deg)
            x_gcn = torch.cat([x, x_gcn], dim=1)
            x_gcn, mis_gcn2, r_gcn_2 = self.rel2_gcn(x_gcn, adj, deg)

            x_gat = torch.cat([x, x_gat], dim=1)
            x_gat, mis_gat2, r_gat_2 = self.rel2_gat(x_gat, adj, deg)

            return torch.cat([x_gcn, x_gat], dim=1), [mis_gcn1, mis_gat1, mis_gcn2, mis_gat2], [r_gcn_1, r_gat_1, r_gcn_2, r_gat_2]
