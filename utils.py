import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch as th
import re
import sys
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
from xml.sax.xmlreader import AttributesNSImpl
import json
from pathlib import Path


def load_config_json(config_path: Path) -> dict:
    with open(config_path) as file:
        return json.load(file)


def encode_input(text, tokenizer, v):
    input = tokenizer(text, max_length=v.max_length, truncation=True,
                      padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask


def get_inputids_attention_mask(v, tokenizer):
    corpse_file = './data/corpus/' + v.dataset + '_shuffle.txt'
    with open(corpse_file, 'r') as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')

    input_ids, attention_mask = encode_input(text, tokenizer, v)
    return input_ids, attention_mask


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# region
# def load_data(dataset_str):
#     """
#     Loads input data from gcn/data directory

#     ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
#         (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
#     ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
#     ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
#     ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
#         object;
#     ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

#     All objects above must be saved using python pickle module.

#     :param dataset_str: Dataset name
#     :return: All data input files loaded (as well the training/test data).
#     """
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))

#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file(
#         "data/ind.{}.test.index".format(dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)
#     print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(
#             min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended

#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#     # print(len(labels))

#     idx_test = test_idx_range.tolist()
#     # print(idx_test)
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y)+500)

#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])

#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]

#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
# endregion

def load_corpus(dataset_str):
    '''
        Loads input corpus from gcn/data directory

        node_size = train_size + test_size + vocab_size
        input: dataset_str
        output: adj -> node_size,node_size (symmetric version created here)
                features -> node_size , word_embed_dim (List of List format sparse)
                y_train -> node_size , 2 (masked array is used to retrieve train part)
                y_val -> node_size , 2
                y_test -> node_size ,2
                train_mask -> node_size,
                val_mask -> node_size,
                test_mask -> node_size,
    '''

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally',
             'adj', 'adj_pmi', 'adj_tfidf', 'adj_nf']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    '''
        real_train_size = int(0.9 * train_size)
        x -> real_train_size, word_embed_dim
        y -> real_train_size, num_class (one-hot vector)
        tx -> test_size, word_embed_dim
        ty -> test_size, num_class
        allx -> train_size + vocab_size , word_embed_dim (including both train and val)
        ally -> train_size + vocab_size , word_embed_dim
    '''
    # use tuple unpacking to list nice implementation
    x, y, tx, ty, allx, ally, adj, adj_pmi, adj_tfidf, adj_nf = tuple(
        objects)
    '''
        features -> node_size, word_embed_dim (List of List format sparse)
        labels -> node_size , num_class
    '''

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    # get train_size,test_size and val_size

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    # define mask arrays for train,test,val
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # make adj matrix symmetric
    adj = make_sym(adj)
    adj_pmi = make_sym(adj_pmi)
    adj_tfidf = make_sym(adj_tfidf)

    return adj, adj_pmi, adj_tfidf, adj_nf, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def make_sym(adj):
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


'''
    new utils functions added by Arda Can Aras
'''


def to_torch_sparse_tensor(M):
    M = M.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((M.row, M.col))).long()
    values = th.from_numpy(M.data)
    shape = th.Size(M.shape)
    T = th.sparse.FloatTensor(indices, values, shape)
    return T


def get_bert_output(dataloader, model, gpu):
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(
                input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    return cls_feat


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize_sparse_graph(graph, gamma, beta):
    """
     Utility function for normalizing sparse graphs.
     return Dr^gamma x graph x Dc^beta
     """
    b_graph = graph.tocsr().copy()
    r_graph = b_graph.copy()
    c_graph = b_graph.copy()
    row_sums = []
    for i in range(graph.shape[0]):
        row_sum = r_graph.data[r_graph.indptr[i]:r_graph.indptr[i+1]].sum()
        if row_sum == 0:
            row_sums.append(0.0)
        else:
            row_sums.append(row_sum**gamma)

    c_graph = c_graph.tocsc()
    col_sums = []
    for i in range(graph.shape[1]):
        col_sum = c_graph.data[c_graph.indptr[i]:c_graph.indptr[i+1]].sum()

        if col_sum == 0:
            col_sums.append(0.0)
        else:
            col_sums.append(col_sum**beta)

    for i in range(graph.shape[0]):
        if row_sums[i] != 0:
            b_graph.data[r_graph.indptr[i]:r_graph.indptr[i+1]] *= row_sums[i]

    b_graph = b_graph.tocsc()
    for i in range(graph.shape[1]):
        if col_sums[i] != 0:
            b_graph.data[c_graph.indptr[i]:c_graph.indptr[i+1]] *= col_sums[i]
    return b_graph


def get_metrics(output, labels, ID=''):
    preds = output.max(1)[1].type_as(labels)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    """Utility function to compute Accuracy, MicroF1 and Macro F1"""
    accuracy = accuracy_score(preds, labels)
    micro = f1_score(preds, labels, average='micro')
    macro = f1_score(preds, labels, average='macro')

    return accuracy, macro, micro
