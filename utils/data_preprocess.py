import random
import networkx as nx
import os
from general import write_pickle, read_pickle


def divide_train_test(edge_s_path, edge_t_path, label_path, new_label_path, tail_node_threshold=10):
    """
    divide dataset to train and test according to degree
    for small dataset
    :param edge_path:
    :param tail_node_threshold: the degree of low-degree nodes of test
    :param label_path: links will be divided into tail/head/super head links
    :param new_label_path: save path of new link dataset
    :return:
    """
    # Gs = nx.read_edgelist(edge_s_path,
    #                      create_using=nx.Graph(), nodetype=int, data=[('weight', int)])
    # Gt = nx.read_edgelist(edge_t_path,
    #                      create_using=nx.Graph(), nodetype=int, data=[('weight', int)])
    adj_s = read_pickle(edge_s_path)
    adj_t = read_pickle(edge_t_path)
    Gs = nx.from_numpy_array(adj_s, create_using=nx.Graph())
    Gt = nx.from_numpy_array(adj_t, create_using=nx.Graph())
    dict_degree_s = dict(nx.degree(Gs))
    dict_degree_t = dict(nx.degree(Gt))

    # links = read_pickle('../data/select/train_test_{:.1f}.pkl'.format(0.5))
    links = read_pickle(label_path)
    all_links = links[0] + links[1]
    dict_all_links = dict(all_links)

    pair_sn = dict_all_links.keys()
    test_links = []
    for sn in pair_sn:
        match_tn = dict_all_links[sn]
        if dict_degree_s[sn] < tail_node_threshold or dict_degree_t[match_tn] < tail_node_threshold:
            test_links.append((sn, match_tn))

    train_links = list(set(all_links).difference(set(test_links)))
    random.shuffle(train_links)
    write_pickle([train_links, test_links], new_label_path)

    print(f'links train: {len(train_links)}, test: {len(test_links)}')

    return train_links, test_links


if __name__ == '__main__':
    # divide dataset into train and test set
    dataset = 'DBLP'
    dataset_dir = f'../../dataset/{dataset}'

    # divide train(head nodes) and test(tail nodes) set according degree
    divide_train_test(os.path.join(dataset_dir, 'adj_s.pkl'),
                      os.path.join(dataset_dir, 'adj_t.pkl'),
                      os.path.join(dataset_dir, 'links_0.5.pkl'),
                      os.path.join(dataset_dir, 'links_degree.pkl'),
                      tail_node_threshold=5)