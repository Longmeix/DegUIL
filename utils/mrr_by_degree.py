from itertools import groupby
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import Tensor
from utils.general import read_pickle
from matplotlib.pyplot import MultipleLocator  # set ticks
import config as cfg
import pandas as pd
from operator import itemgetter

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})  # 改变所有字体大小，改变其他性质类似

"""
analyse how node degree impacts performance
conclusion: the node degree decreases, the performance also decreases 
idea from: CIKM20 Towards locality-aware meta-learning of tail node embeddings on networks 
"""


def cosine(xs, ys, epsilon=1e-8):
    """
    Efficiently calculate the pairwise cosine similarities between two set of vectors.
    Args:
        xs: feature matrix, [N, dim]
        ys: feature matrix, [M, dim]
        epsilon: a small number to avoid dividing by zero
    Return:
        a [N, M] matrix of pairwise cosine similarities
    """
    mat = xs @ ys.t()
    x_norm = xs.norm(2, dim=1) + epsilon
    y_norm = ys.norm(2, dim=1) + epsilon
    x_diag = (1 / x_norm).diag()
    y_diag = (1 / y_norm).diag()
    return x_diag @ mat @ y_diag


def get_metrics(emb_s, emb_t, k, links, is_test=True, default=0.):
    """
    Calculate the average precision@k and hit_precision@k from two sides, i.e., source-to-target and target-to-source.
    Args:
      sims_mat: similarity matrix
      k: number of candidates
      link: index pairs of matched users, i.e., the ground truth
    Return:
      coverage: precision@k
      hit: hit_precison@k
    """
    # row: source node  col: target node
    train_link, test_link = links
    row, col = [list(i) for i in zip(*train_link)]
    sims_mat = cosine(emb_s, emb_t)
    # mask similarities of matched user pairs in the train set
    if is_test:
        with torch.no_grad():
            sims_mat[row] = default
            sims_mat[:, col] = default
    row, col = [list(i) for i in zip(*test_link)]
    target = sims_mat[row, col].reshape((-1, 1))  # s,t两用户节点真实link的相似度
    s_sim = sims_mat[row]
    t_sim = sims_mat.t()[col]
    # match users from source to target
    mrr_s, hit_s = score(s_sim, target, k)
    print('Test MRR: {:.4f}, Hit: {:.4f}'.format(mrr_s.mean().item(), hit_s.mean().item()))
    mrr_s_result = dict(zip(*(row, mrr_s.detach().cpu().numpy())))
    hit_s_result = dict(zip(*(row, hit_s.detach().cpu().numpy())))
    # # match users from target to source
    # c_t, h_t, _ = score(t_sim, target, k)
    # averaging the scores from both sides
    # return (c_s + c_t) / 2, (h_s + h_t) / 2
    return mrr_s_result, hit_s_result


def score(sims: Tensor, target: Tensor, k: int) -> tuple:
    """
    Calculate the average precision@k and hit_precision@k from while matching users from the source network to the target network.
    Args:
        sims: similarity matrix
        k: number of candidates
        test: index pairs of matched users, i.e., the ground truth
    Return:
        coverage: precision@k
        hit: hit_precison@k
    """
    # h(x) = (k - (hit(x) - 1) / k  review 4.2节的公式
    # number of users with similarities larger than the matched users
    rank = (sims >= target).sum(1)
    # rank = min(rank, k + 1)
    mrr = (1. / rank)
    rank = rank.min(torch.tensor(k + 1).to(sims.device))
    idx_fail = torch.nonzero(rank == (k + 1)).squeeze()  # get node indexes fail to link

    temp = (k + 1 - rank).float()
    # hit_precision@k
    hit_p = temp / k
    # precision@k
    # coverage = (temp > 0).float().mean()
    return mrr, hit_p


def filter_links_by_degree_interval(degree_list, test_links, intervals):
    test_s_nodes, test_t_nodes = [list(i) for i in list(zip(*test_links))]
    test_s_degree = pd.DataFrame(degree_list, columns=['idx', 'degree']).loc[test_s_nodes]

    dict_deg_idxes = dict()
    criteria = pd.cut(test_s_degree['degree'], bins=intervals)

    for internal, group in test_s_degree.groupby(criteria):
        dict_deg_idxes[f'{internal.left}-{internal.right}'] = list(group['idx'])

    for internal, idxes in dict_deg_idxes.items():
        print('{}: {}'.format(internal, len(idxes)))

    return dict_deg_idxes

def hit_rank_by_degree(links, embs, degree_list, intervals, is_test, k, degree_len=10):
    emb_s, emb_t = embs

    dict_deg_idxes = filter_links_by_degree_interval(degree_list, links[1], intervals)
    mrr_test_list, hit_test_list = get_metrics(emb_s, emb_t, k, links, is_test)

    intervals_columns, num_links = [], []
    mrr_bins, hit_bins = [], []
    for interval, idxes in dict_deg_idxes.items():
        intervals_columns.append(interval)
        num_links.append(len(idxes))
        if len(idxes) == 0:
            mrr_bins.append(0)
            hit_bins.append(0)
            continue

        mrr_bin = itemgetter(*idxes)(mrr_test_list)
        hit_bin = itemgetter(*idxes)(hit_test_list)
        mrr = np.mean(mrr_bin)
        hit = np.mean(hit_bin)
        mrr_bins.append(mrr)
        hit_bins.append(hit)
        print('{}: MRR={:.4f}, Hit={:.4f}'.format(interval, mrr, hit))
    return mrr_bins


def plot_result_by_degree(dataset, results_bins, intervals):
    plt.figure(figsize=(6, 5))
    width = 3
    x = [i * width for i in range(len(results_bins))]
    plt.bar(x, results_bins, width=width, color='#82B0D2', edgecolor='black', linewidth=0.7)
    # the_table = plt.table(cellText=[intervals_columns, num_links],
    #                       rowLabels=['degrees', '# links'],
    #                       cellLoc='center',
    #                       rowLoc='center',
    #                       loc='bottom')
    # the_table.set_fontsize(15)

    # plt.subplots_adjust(left=0.2, bottom=0.5)
    # plt.xticks(np.array(x) - width/2, intervals[:-1])
    plt.xticks(np.array([i * width for i in range(len(intervals))]) - width / 2, intervals, fontsize=20)
    plt.yticks(np.linspace(0, 0.6, 4), fontsize=20)
    # plt.ylim([0, 1])
    plt.xlabel("Node degree")
    plt.ylabel("MRR")
    # plt.label([])

    plt.tight_layout(pad=0.4, h_pad=None, w_pad=None, rect=None)
    plt.savefig(dataset + '_mrr_by_degree.pdf', bbox_inches='tight')
    plt.show()


def plot_result_by_degree_compare(dataset, res_poor, res_good, intervals):
    width = 2
    better = np.array(res_good) - np.array(res_poor)
    x = [i * width for i in range(len(res_poor))]
    plt.bar(x, res_poor, width=width, color='#82B0D2', edgecolor='black', linewidth=0.9)
    plt.bar(x, better, width=width, color='#F0988C', edgecolor='black', linewidth=0.9, bottom=res_poor)

    plt.xticks(np.array([i * width for i in range(len(intervals))]) - width / 2, intervals, fontsize=22)
    # plt.ylim([0, 1])
    plt.yticks(np.linspace(0, 1, 6), fontsize=22)
    plt.xlabel("Node degree")
    plt.ylabel("MRR")
    plt.title(dataset, fontsize=24)

    plt.tight_layout(pad=0.4, h_pad=None, w_pad=None, rect=None)
    # plt.savefig(name + '.pdf', bbox_inches='tight')
    # plt.show()

def dataset_reslust_by_degree(dataset):
    # dataset = 'DBLP'  # FT, DBWB
    adj = read_pickle(f'../../dataset/{dataset}/adj_s.pkl')
    adj = ((adj + adj.T) > 0) * 1.
    G = nx.from_numpy_matrix(adj, create_using=nx.Graph())
    deg = nx.degree(G)
    degree_list = list(deg)

    # impact of node degree on prediction
    embs_pale = read_pickle(f'../../dataset/models/PALE/best_embs_{dataset}_0.5.pkl')
    embs_deguil = read_pickle(f'../../DegUIL/datasets/bestEmbs/DegUIL_{dataset}_0.5_best_embs.pkl')
    links = read_pickle(f'../../dataset/{dataset}/links_0.5.pkl')

    deg_max = max(np.array(degree_list)[:, 1])
    intervals = []
    if dataset == 'FT':
        intervals = [0, 1, 3, 5, 10, 20, 50, 80, 100, 200, deg_max]
    elif dataset == 'DBLP':
        intervals = [0, 2, 5, 10, 20, 50, 70, 90, 100, 110, deg_max]
    else:
        print('Please set degree interval for dataset')
    # get results by degree
    mrr_pale = hit_rank_by_degree(links, embs_pale, degree_list, intervals, is_test='test', k=30)
    mrr_deguil = hit_rank_by_degree(links, embs_deguil, degree_list, intervals, is_test='test', k=30)
    # plot stacked bar for comparison
    plot_result_by_degree_compare(dataset, mrr_pale, mrr_deguil, intervals)


def plot_degree_distribution(G):
    # 返回图中所有节点的度分布序列
    degree_count = nx.degree_histogram(G)
    # 生成x轴序列，从1到最大度
    degree = range(len(degree_count))
    # # 将频次转换为频率
    # y = [z / float(sum(degree_count)) for z in degree_count]
    # plt.figure(1)
    plt.figure(figsize=(6, 5))
    # # plt.title('Douban-Weibo')
    plt.xlabel('Node degree')
    plt.ylabel('Number of nodes')
    # # 在双对数坐标轴上绘制度分布曲线
    plt.loglog(degree, degree_count, '.', markersize=10, color="#4a8cbe")
    # plt.plot(degree, y, '.', color="#0072BD")
    plt.tight_layout(pad=0.4, h_pad=None, w_pad=None, rect=None)
    plt.savefig('long_tailed.pdf', bbox_inches='tight')
    plt.show()


    # # plt.scatter(x, y, s=1, color=(1,0,0))
    # # plt.title('Pow-law')
    # plt.xlabel('Node degree')
    # plt.ylabel('Number of nodes')
    # plt.scatter(degree, degree_count, s=1, color=(1, 0, 0))
    # plt.tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    # plt.savefig('pow_law2.pdf', bbox_inches='tight')
    # # 显示图表
    # plt.show()

if __name__ == '__main__':
    # nodes degree
    dataset = 'FT'  # FT, DBWB
    adj = read_pickle(f'../../dataset/{dataset}/adj_s.pkl')
    adj = ((adj + adj.T) > 0) * 1.
    G = nx.from_numpy_matrix(adj, create_using=nx.Graph())

    deg = nx.degree(G)
    degree_list = list(deg)

    # impact of node degree on prediction
    embs_pale = read_pickle(f'../dataset/models/PALE/best_embs_{dataset}_0.5.pkl')
    embs_deguil = read_pickle(f'../datasets/bestEmbs/DegUIL_{dataset}_0.5_best_embs.pkl')
    links = read_pickle(f'../../dataset/{dataset}/links_0.5.pkl')

    deg_max = max(np.array(degree_list)[:, 1])
    intervals = []
    if dataset == 'FT':
        intervals = [0, 1, 3, 5, 10, 20, 50, 80, 100, 200, deg_max]
    elif dataset == 'DBLP':
        intervals = [0, 2, 5, 10, 20, 50, 70, 90, 100, 110, deg_max]
    else:
        print('Please set degree interval for dataset')
    # get results by degree
    mrr_pale = hit_rank_by_degree(links, embs_pale, degree_list, intervals, is_test='test', k=30)
    mrr_deguil = hit_rank_by_degree(links, embs_deguil, degree_list, intervals, is_test='test', k=30)

    '''plot in introduction'''
    # plot_degree_distribution(G)
    # plot_result_by_degree(dataset, mrr_pale, intervals)

    '''plot stacked bar for experimental comparison'''
    plot_result_by_degree_compare(dataset, mrr_pale, mrr_deguil, intervals)
