import config as cfg
from models.base import UIL
from scipy.sparse import csr_matrix
import torch
from torch import nn
import torch.optim as opt
from utils.general import write_pickle, read_pickle
from models.loss_neg import NSLoss
import numpy as np
from itertools import chain
import utils.extensions as npe

net_key = ['s', 't']


class MLPmap(UIL):
    def __init__(self, embeds, links, adj_t, args):
        super(MLPmap, self).__init__(links, k=args.top_k)
        self.device = args.device
        dim = args.embed_dim
        self.adj_t = adj_t
        # transfer embeds type to tensor
        if isinstance(embeds[0], torch.Tensor):
            self.embeds = dict(zip(net_key, (emb.to(self.device) for emb in embeds)))
            # self.embeds = dict(zip(net_key, (F.normalize(emb, dim=1, p=2).to(self.device) for emb in embeds)))
        else:
            self.embeds = dict(zip(net_key, (torch.tensor(emb).float().to(self.device) for emb in embeds)))

        self.mapping = nn.ModuleList([
            MLP(dim, cfg.dim_feature),
            MLP(dim, cfg.dim_feature)]).to(self.device)

        # s, t = list(zip(*links[0]))
        # self.link_mat = csr_matrix((np.ones(len(s)), (s, t)),
        #                            shape=(embeds[0].shape[0], embeds[1].shape[0]))
        shape = embeds[0].shape[0], embeds[1].shape[0]
        link_train, link_test = links
        link_mat = npe.pair2sparse(link_train, shape)
        self.link_attr_train = self.addEdgeAttr(link_mat)  # real links of train set
        self.pairs_l, self.wgt_l, self.adj_l = self.link_attr_train.values()  # link match

        self.opt_map = opt.Adam(
            chain(self.mapping.parameters(),
                  ),
            lr=args.mapping_lr,
            # weight_decay=1e-4
        )

        self.loss_label = NSLoss(
            sim=nn.CosineSimilarity(),
            # act=nn.ReLU(),
            loss=nn.MSELoss(reduction='sum')
        )
        self.mse = nn.MSELoss()

    @staticmethod
    def addEdgeAttr(adj_mat, exponent=3 / 4, percent=cfg.percent):
        if not isinstance(adj_mat, np.ndarray):
            adj_mat = adj_mat.toarray()
        weights = np.abs(adj_mat > 0).sum(axis=0) ** exponent
        clamp = npe.clamp_mat(adj_mat, percent)
        pairs = [i.tolist() for i in clamp.nonzero()]
        pairs = list(zip(*pairs))

        attr_keys = ['pairs', 'weights', 'adj_mat']
        attr_value = pairs, weights, csr_matrix(adj_mat)
        return dict(zip(attr_keys, attr_value))

    def get_sims(self):
        f_s, f_t = self.get_embeds(is_map=True)
        sims = self.sim_pairwise(f_s, f_t)
        return sims

    def match_loss(self, pair):
        # idx_s, idx_t = [list(i) for i in list(zip(*pair))]
        idx_s, idx_t = pair
        embed_s, embed_t = self.get_embeds(is_map=False)

        loss_batch = self.loss_label(
            embed_s, embed_t,
            idx_s, idx_t,
            self.mapping[0],
            self.mapping[1],
            # lambda x: x,
            cfg.neg_num,
            # None,  # None if sample=='neg'
            self.wgt_l,
            self.adj_l,
            self.adj_t,
            'deg'
        )
        return loss_batch

    def train_labels_neg(self):
        batches = self.load_data(
            self.links[0], cfg.batch_size)
        N = len(batches)
        loss_c = 0.
        for l_pair in batches:
            loss = self.match_loss(l_pair)
            loss_c += loss
            self.optimize(self.opt_map, loss)
        loss_c /= N
        return loss_c

    def train(self):
        mrr_best = 0.
        for epoch in range(1, cfg.epochs + 1):
            loss = self.train_labels_neg()
            self.log.info('epoch {:03d} loss_labels {:.4f}'.format(
                epoch, loss))

            # eval
            with torch.no_grad():
                embed_s, embed_t = self.get_embeds(is_map=True)
                mrr, hit_p = self.eval_hit_p(embed_s, embed_t, self.k, default=-1.)
            # flag = self.early_stop(hit_p, predicts)
            if mrr > mrr_best:
                mrr_best = mrr
                write_pickle([embed_s.detach().cpu(), embed_t.detach().cpu()], cfg.best_embs_file)
                # self.save_emb('data/{}/model_{:.1f}.pkl'.format(cfg.model, cfg.ratio))
            self.log.info('Epoch: {}, MRR_best: {:.4f}'.format(epoch, mrr_best))

        emb_s, emb_t = read_pickle(cfg.best_embs_file)
        self.report_hit(emb_s, emb_t)
        self.rename_log('/mrr{:.2f}_'.format(mrr_best * 100).join(
            cfg.log.split('/')
        ))


    def get_embeds(self, is_map=False):
        # get node embedding of two networks
        # @is_map: map source embeddings or not
        embed_s, embed_t = self.embeds['s'], self.embeds['t']
        if is_map:
            embed_s = self.mapping[0](embed_s)
            embed_t = self.mapping[1](embed_t)
        return embed_s, embed_t


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
            nn.Linear(dim_in, dim_hid),
            # nn.Dropout(0.5),
            act,
            nn.Linear(dim_hid, dim_out)
            # nn.Linear(dim_in, dim_out)
        )

    def forward(self, x):
        return self.model(x)
