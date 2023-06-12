import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange
import config as cfg
import utils.extensions as npe
from models.loss_neg import NSLoss
from models.base import DataWrapper
from models.base import UIL
from utils.general import write_pickle, read_pickle

net_key = ['s', 't']


class UILAggregator(UIL):
    def __init__(self, adj_s, adj_t, links, k):
        super(UILAggregator, self).__init__(links, k=k)
        adj_s = csr_matrix(adj_s.cpu().numpy())
        adj_t = csr_matrix(adj_t.cpu().numpy())
        self.edgeAttr = self.getEdgeAttr(adj_s, adj_t)
        shape = adj_s.shape[0], adj_t.shape[0]

        if links is not None:
            link_train, link_test = links
            link_mat = npe.pair2sparse(link_train, shape)
            self.link_attr_train = self.addEdgeAttr(link_mat)  # real links of train set
        # s,t两个网络中 每一列是一条表边(vi,vj) edge['s'].size (2, E)
        self.edge_idx = dict(s=self.adj2edge(adj_s), t=self.adj2edge(adj_t))

        self.pairs_s, self.wgt_s, self.adj_s = self.edgeAttr['s'].values()  # source net
        self.pairs_t, self.wgt_t, self.adj_t = self.edgeAttr['t'].values()  # target net
        self.pairs_l, self.wgt_l, self.adj_l = self.link_attr_train.values()  # link match

        self.hit_best, self.mrr_best = .0, .0

        # loss
        self.loss_intra = NSLoss(
            act=nn.Sigmoid(),
            sim=nn.CosineSimilarity(),
            loss=nn.BCEWithLogitsLoss()
        )

        self.loss_cosine = NSLoss(
            sim=nn.CosineSimilarity(),
            loss=nn.MSELoss(reduction='sum')
        )
        self.mse = nn.MSELoss(reduction='mean')

    @staticmethod
    def addEdgeAttr(adj_mat, exponent=3 / 4, percent=cfg.percent):
        """
       Given a similarity matrix, create weights for negative sampling and indices of similar users.
       Args:
           mat: similarity matrix
           exponent: a coefficient to downplay the similarities to create negative sampling weights, default: 3/4 (as suggested by word2vec)
           percent: percent of users to filter, range in [0, 100]
       Return:
           pairs: user pairs with high similairties
           weights: negative sampling weights
           mat: similarity matrix
       """
        if not isinstance(adj_mat, np.ndarray):
            adj_mat = adj_mat.toarray()
        weights = np.abs(adj_mat > 0).sum(axis=0) ** exponent
        clamp = npe.clamp_mat(adj_mat, percent)
        pairs = [i.tolist() for i in clamp.nonzero()]
        pairs = list(zip(*pairs))

        attr_keys = ['pairs', 'weights', 'adj_mat']
        attr_value = pairs, weights, csr_matrix(adj_mat)
        return dict(zip(attr_keys, attr_value))

    @staticmethod
    def adj2edge(adj_mat):
        # get edge(vi, vj) from adjacent matrix
        # size (2, E)
        return torch.tensor(list(zip(*adj_mat.nonzero()))).long().t().to(cfg.device)

    def getEdgeAttr(self, adj_s, adj_t):
        # get [pair, weight, adj_mat] of network s and t
        edgeAttr = {'s': {}, 't': {}}
        edgeAttr['s'] = self.addEdgeAttr(adj_s)
        edgeAttr['t'] = self.addEdgeAttr(adj_t)
        return edgeAttr

    def get_sims(self):
        f_s, f_t = self.get_embeds(is_eval=True)
        sims = self.sim_pairwise(f_s, f_t)
        return sims

    @staticmethod
    def get_pair_batch(pairs, batch_size):
        '''
        @:pairs: sample pairs[(vs, vt), ...], size=cfg.batch_size
        @:return: batch [(vs1, vs2, ...), (vt1, ...)]
        '''
        idx = RandomSampler(pairs, replacement=True,
                            num_samples=batch_size)
        pair_list = [pairs[i] for i in list(idx)]
        data = DataWrapper(pair_list)
        batches = DataLoader(
            data, batch_size=batch_size,
            shuffle=True)
        _, batch = next(enumerate(batches))  # only one batch in batches
        return batch

    def global_loss(self, embed, pair, weight, adj_mat):
        idx_s, idx_t = pair
        loss_batch = self.loss_cosine(
            embed, embed,
            idx_s, idx_t,
            lambda x: x,
            lambda x: x,
            cfg.neg_num,
            weight,
            adj_mat,
            adj_mat,
            'deg'
        )

        return loss_batch

    def local_loss(self, embed, pair):
        idx_s, idx_t = pair
        loss_batch = self.mse(embed[idx_s], embed[idx_t])
        return loss_batch

    def embed_map_loss(self, embed_s, embed_t, l_pair, maps):
        idx_s, idx_t = l_pair
        source_emb, target_emb = embed_s[idx_s], embed_t[idx_t]
        source_emb_after_map = F.normalize(maps[0](source_emb))
        target_emb_after_map = F.normalize(maps[1](target_emb))
        loss_st = self.mse(source_emb_after_map, target_emb)
        loss_ts = self.mse(target_emb_after_map, source_emb)
        return loss_st + loss_ts

    def match_loss(self, embed_s, embed_t, common, pair, weight, adj_mat, adj_t):
        idx_s, idx_t = pair

        loss_batch = self.loss_cosine(
            embed_s, embed_t,
            idx_s, idx_t,
            common[0],
            common[1],
            cfg.neg_num,
            weight,
            adj_mat,
            adj_t,
            'deg'
        )
        return loss_batch

    def unsupervised_loss(self, s_pair, t_pair, l_pair, embed_s, embed_t, common):
        unlabel_and_label_s_node = torch.cat([s_pair[0], s_pair[0], l_pair[0]])
        unlabel_and_label_t_node = torch.cat([t_pair[0], t_pair[1], l_pair[1]])
        emb_sn = embed_s[list(unlabel_and_label_s_node)]
        emb_tn = embed_t[list(unlabel_and_label_t_node)]
        semi_map_ss = common[1](common[0](emb_sn))
        semi_map_tt = common[0](common[1](emb_tn))
        map_loss_ss = self.mse(emb_sn, semi_map_ss)
        map_loss_tt = self.mse(emb_tn, semi_map_tt)
        return map_loss_ss + map_loss_tt

    def train(self, embed_s, embed_t, common):
        # get batch data
        s_pair = self.get_pair_batch(self.pairs_s, cfg.batch_size)
        t_pair = self.get_pair_batch(self.pairs_t, cfg.batch_size)
        l_pair = self.get_pair_batch(self.pairs_l, cfg.batch_size)

        # ========= global loss ==========
        loss_g_s = self.global_loss(embed_s, s_pair, self.wgt_s, self.adj_s)
        loss_g_t = self.global_loss(embed_t, t_pair, self.wgt_t, self.adj_t)
        loss_global = loss_g_s + loss_g_t

        loss_match = self.match_loss(embed_s, embed_t, common, l_pair, self.wgt_l, self.adj_l, self.adj_t)

        # sum all loss
        loss_batch = 0.2 * loss_global + loss_match

        return loss_batch

    # ======= evaluate ==========
    def eval_hit(self, epoch, embed_s, embed_t, common):
        embed_s, embed_t = common[0](embed_s), common[1](embed_t)

        mrr, hit_p = self.eval_hit_p(embed_s, embed_t, self.k, default=-1.)
        if mrr > self.mrr_best:
            self.mrr_best = mrr
            self.hit_best = hit_p
            if epoch > 400:  # save best model, saving time
                write_pickle([embed_s.detach().cpu(), embed_t.detach().cpu()], cfg.best_embs_file)
        self.log.info('Epoch: {}, MRR_best: {:.4f}, Hit_best: {:.4f}'.format(epoch, self.mrr_best, self.hit_best))

    def print_performance_k(self):
        emb_s, emb_t = read_pickle(cfg.best_embs_file)
        self.report_hit(emb_s, emb_t)
        self.rename_log('/mrr{:.2f}_'.format(self.mrr_best * 100).join(
            cfg.log.split('/')
        ))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out,
                 dim_hid=None, act=None):
        super(MLP, self).__init__()
        if act is None:
            act = nn.Tanh()
        if dim_hid is None:
            dim_hid = dim_in * 2
        # 2-layers
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_hid, bias=True),
            act,
            nn.Linear(dim_hid, dim_out, bias=True)
        )

    def forward(self, x):
        return self.model(x)
