import os

import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_Mutagenicity_data(path="Mutag/", dataset="Mutag", split_train=0.7, split_val=0.15):
    """Load Mutagenicity data """
    print('Loading {} dataset...'.format(dataset))

    nodeidx_features = np.genfromtxt("{}{}.node_labels".format(path, dataset), delimiter=",",
                                     dtype=np.dtype(int))
    features = np.zeros((nodeidx_features.shape[0], max(nodeidx_features) + 1))
    features[np.arange(nodeidx_features.shape[0]), nodeidx_features] = 1
    features = sp.csr_matrix(features, dtype=np.float32)

    labels = np.genfromtxt("{}{}.graph_labels".format(path, dataset),
                           dtype=np.dtype(int))
    labels = encode_onehot(labels)

    graph_idx = np.genfromtxt("{}{}.graph_idx".format(path, dataset),
                              dtype=np.dtype(int))
    graph_idx = np.array(graph_idx, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(graph_idx)}

    edges_unordered = np.genfromtxt("{}{}.edges".format(path, dataset), delimiter=",",
                                    dtype=np.int32)
    edges_label = np.genfromtxt("{}{}.link_labels".format(path, dataset), delimiter=",",
                                dtype=np.int32)

    # According to paper, ignore edge labels
    # adj = sp.coo_matrix((edges_label, (edges_unordered[:,0]-1, edges_unordered[:,1]-1)))
    adj = sp.coo_matrix((np.ones(len(edges_label)), (edges_unordered[:, 0] - 1, edges_unordered[:, 1] - 1)))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    num_total = max(graph_idx)
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)

    if (num_train == num_val or num_val == num_total):
        return

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_map, idx_train, idx_val, idx_test


def load_MUTAG_data(path="MUTAG/", dataset="MUTAG_", split_train=0.7, split_val=0.15):
    """Load MUTAG data """
    print('Loading {} dataset...'.format(dataset))

    nodeidx_features = np.genfromtxt("{}{}node_labels.txt".format(path, dataset), delimiter=",",
                                     dtype=np.dtype(int))
    features = np.zeros((nodeidx_features.shape[0], max(nodeidx_features) + 1))
    features[np.arange(nodeidx_features.shape[0]), nodeidx_features] = 1
    features = sp.csr_matrix(features, dtype=np.float32)

    labels = np.genfromtxt("{}{}graph_labels.txt".format(path, dataset),
                           dtype=np.dtype(int))
    labels = encode_onehot(labels)

    graph_idx = np.genfromtxt("{}{}graph_indicator.txt".format(path, dataset),
                              dtype=np.dtype(int))
    graph_idx = np.array(graph_idx, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(graph_idx)}

    edges_unordered = np.genfromtxt("{}{}A.txt".format(path, dataset), delimiter=",",
                                    dtype=np.int32)
    edges_label = np.genfromtxt("{}{}edge_labels.txt".format(path, dataset), delimiter=",",
                                dtype=np.int32)
    adj = sp.coo_matrix((edges_label, (edges_unordered[:, 0] - 1, edges_unordered[:, 1] - 1)))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    num_total = max(graph_idx)
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)

    if (num_train == num_val or num_val == num_total):
        return

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_map, idx_train, idx_val, idx_test


def load_split_MUTAG_data(path="MUTAG/", dataset="MUTAG_", split_train=0.7, split_val=0.15):
    """Load MUTAG data """
    print('Loading {} dataset...'.format(dataset))

    labels = np.genfromtxt(os.path.join(path, "{}graph_labels.txt".format(dataset)),
                           dtype=np.dtype(int))

    labels = encode_onehot(labels)
    labels = torch.LongTensor(np.where(labels)[1])

    graph_idx = np.genfromtxt("{}{}graph_indicator.txt".format(path, dataset),
                              dtype=np.dtype(int))
    graph_idx = np.array(graph_idx, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(graph_idx)}
    length = len(idx_map.keys())
    num_nodes = [idx_map[n] - idx_map[n - 1] if n - 1 > 1 else idx_map[n] for n in range(1, length + 1)]
    max_num_nodes = max(num_nodes)
    features_list = []
    adj_list = []
    prev = 0

    nodeidx_features = np.genfromtxt("{}{}node_labels.txt".format(path, dataset), delimiter=",",
                                     dtype=np.dtype(int))
    features = np.zeros((nodeidx_features.shape[0], max(nodeidx_features) + 1))
    features[np.arange(nodeidx_features.shape[0]), nodeidx_features] = 1

    edges_unordered = np.genfromtxt("{}{}A.txt".format(path, dataset), delimiter=",",
                                    dtype=np.int32)
    edges_label = np.genfromtxt("{}{}edge_labels.txt".format(path, dataset), delimiter=",",
                                dtype=np.int32)
    adj = sp.coo_matrix((edges_label, (edges_unordered[:, 0] - 1, edges_unordered[:, 1] - 1)))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj.todense()

    for n in range(1, length + 1):
        entry = np.zeros((max_num_nodes, max(nodeidx_features) + 1))
        entry[:idx_map[n] - prev] = features[prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        features_list.append(entry)

        entry = np.zeros((max_num_nodes, max_num_nodes))
        entry[:idx_map[n] - prev, :idx_map[n] - prev] = adj[prev:idx_map[n], prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        adj_list.append(entry)

        prev = idx_map[n]

    num_total = max(graph_idx)
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)

    if (num_train == num_val or num_val == num_total):
        return

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_list, features_list, labels, idx_map, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
