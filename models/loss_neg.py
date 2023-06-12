import torch
from torch import Tensor
import torch.nn as nn
import config as cfg
from utils.extensions import cosine as pairwise_cosine

class NSLoss(nn.Module):
    '''negetive sampling loss'''
    def __init__(self, sim=None, act=None, loss=None):
        """
        Args:
            sim: a similarity function
            act: a activator function that map the similarity to a valid domain
            loss: a criterion measuring the predicted similarities and the ground truth labels
        """
        super(NSLoss, self).__init__()
        if sim is not None:
            self.sim = sim
        else:
            self.sim = self.inner_product
        if act is not None:
            self.act = act
        else:
            self.act = self.identity
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.MSELoss()
        # self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.2)
        # self.triple_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    @staticmethod
    def inner_product(x, y):
        return x.mul(y).sum(dim=1, keepdim=True)

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def get_adjacent(idx_s: Tensor, idx_t: Tensor, adj_mat) ->Tensor:
        """
        Given indices, get corresponding weights from a weighted adjacency matrix.
        Args:
            i_s: row indices
            i_t: column indices
            weights: a weighted adjacency matrix, a sparse matrix is preferred as it saves memory
        Return:
            a weight vector of length len(i_s)
        """
        if adj_mat is None:
            return torch.ones(len(idx_s))
        else:
            idx_s = idx_s.tolist()
            idx_t = idx_t.tolist()
            adjacent = adj_mat[idx_s, idx_t]
            return torch.FloatTensor(adjacent).squeeze()

    @staticmethod
    def sample(neg_num: int, idx_t: Tensor, adj_t, probs, sim, sample='neg'):
        assert neg_num > 0
        batch_size = len(idx_t)
        # adj_t = ((adj_t + adj_t.T) > 0) * 1.

        if not isinstance(adj_t, torch.Tensor):
            adj_t = torch.tensor(adj_t.toarray())
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)

        if sample == 'deg':
            idx = torch.multinomial(probs, batch_size * neg_num, True)  # sample hub nodes as negative examples
        elif sample == 'neg':
            idx = torch.Tensor(batch_size * neg_num).uniform_(0, len(adj_t)).long()  # random uniformly
        elif sample == 'sim':  # top-k closest nodes
            idx_t_sim = sim[idx_t]
            # value, idx = torch.topk(idx_t_sim, k=neg_num, dim=-1, largest=True)
            idx = torch.multinomial(torch.sigmoid(idx_t_sim), neg_num, True)
            idx = idx.t()
        return idx.view(neg_num, batch_size)

    def get_xy(self, *input):
        embed_s, embed_t, idx_s, idx_t, map_s, map_t,\
            neg_num, probs, adj_mat, adj_t, sample_way = input
        embed_s, embed_t = map_s(embed_s), map_t(embed_t)
        similarity = pairwise_cosine(embed_s, embed_t)
        x_s, x_t = embed_s[idx_s], embed_t[idx_t] # positive
        # calculate node similarities, stand for positive sample
        # x_pos = x_t
        y_pos = self.get_adjacent(idx_s, idx_t, adj_mat)

        if neg_num > 0:
            batch_size = len(idx_s)
            idx_neg = self.sample(neg_num, idx_t, adj_t, probs, similarity, sample_way)
            x_neg = torch.stack([embed_t[idx] for idx in idx_neg])
            y_neg_st = torch.stack([
                        self.get_adjacent(idx_s, idx, adj_mat) for idx in idx_neg
                    ]).view(-1)  # negative sample for s to t

            y_neg_tt = torch.stack([
                        self.get_adjacent(idx_t, idx, adj_t) for idx in idx_neg
                    ]).view(-1)  # negative sample for t to t

        else:
            # x_t = x_pos
            y_pos = y_pos

        return x_s, x_t, x_neg, y_pos, y_neg_st, y_neg_tt

    def forward(self, *input):
        x_s, x_t, x_neg, y_pos, y_neg_st, y_neg_tt = self.get_xy(*input)
        y_hat = self.act(self.sim(x_s, x_t))
        if y_hat.is_cuda:
            y_pos = y_pos.to(cfg.device)
            y_neg_st = y_neg_st.to(cfg.device)
            y_neg_tt = y_neg_tt.to(cfg.device)
        # 1. positive loss
        loss_pos = self.loss(y_hat, y_pos)
        # 2. negative loss of s to t
        y_hat_st = self.act(torch.stack([self.sim(x_s, x) for x in x_neg
                             ]).view(-1))
        loss_neg_st = self.loss(y_hat_st, y_neg_st)
        # 3. negative loss of target to negative nodes in network t
        y_hat_tt = self.act(torch.stack([self.sim(x_t, x) for x in x_neg
                                         ]).view(-1))
        loss_neg_tt = self.loss(y_hat_tt, y_neg_tt)

        # return loss_pos
        return (loss_pos + loss_neg_st + loss_neg_tt) / len(y_hat) # - loss_neg_tt
        # return loss_pos + loss_neg_st + loss_neg_tt

    # def get_xy(self, *input):
    #     embed_s, embed_t, idx_s, idx_t, map_s, map_t,\
    #         neg_num, probs, adj_mat, adj_t, sample_way = input
    #     embed_s, embed_t = map_s(embed_s), map_t(embed_t)
    #     similarity = pairwise_cosine(embed_s, embed_t)
    #     x_s, x_t = embed_s[idx_s], embed_t[idx_t] # positive
    #     # x_s, x_t = map_s(x_s), map_t(x_t)
    #     # calculate node similarities, stand for positive sample
    #     # x_pos = x_t
    #     y_pos = self.get_adjacent(idx_s, idx_t, adj_mat)
    #
    #     if neg_num > 0:
    #         batch_size = len(idx_s)
    #         idx_neg = self.sample(neg_num, idx_t, adj_t, probs, similarity, sample_way)
    #         x_neg = torch.stack([embed_t[idx] for idx in idx_neg])
    #         y_neg_st = torch.stack([
    #                     self.get_adjacent(idx_s, idx, adj_mat) for idx in idx_neg
    #                 ])  # negative sample for s to t
    #
    #         y_neg_tt = torch.stack([
    #                     self.get_adjacent(idx_t, idx, adj_t) for idx in idx_neg
    #                 ])  # negative sample for t to t
    #
    #     else:
    #         # x_t = x_pos
    #         y_pos = y_pos
    #
    #     return x_s, x_t, x_neg, y_pos, y_neg_st*2 - 1, y_neg_tt*2 - 1
    #
    # def forward(self, *input):
    #     x_s, x_t, x_neg, y_pos, y_neg_st, y_neg_tt = self.get_xy(*input)
    #     y_hat = self.sim(x_s, x_t)
    #     if y_hat.is_cuda:
    #         y_pos = y_pos.to(cfg.device)
    #         y_neg_st = y_neg_st.to(cfg.device)
    #         y_neg_tt = y_neg_tt.to(cfg.device)
    #
    #     # loss_s = self.triple_loss()
    #     loss_pos = self.cosine_loss(x_s, x_t, y_pos)
    #     loss_neg_st, loss_neg_tt = .0, .0
    #     for i, x_n in enumerate(x_neg):
    #         # loss_neg_st += self.triple_loss(x_s, x_t, x_n)
    #         loss_neg_st += self.cosine_loss(x_s, x_n, y_neg_st[i])
    #         loss_neg_tt += self.cosine_loss(x_t, x_n, y_neg_tt[i])
    #
    #     return loss_pos + loss_neg_st + loss_neg_tt
    #     # return loss_neg_st