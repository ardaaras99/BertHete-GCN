# %%
import random
from re import S
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from math import log

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from types import SimpleNamespace
from pathlib import Path
from utils import *
'''
    data/corpus -> directory has raw text of the documents
                    each line is training instance

    data/       -> directory has txt file that stores each data
                    instance id, train or test identification and
                    label
'''
WORK_DIR = Path(__file__).parent
CONFIG_PATH = Path.joinpath(
    WORK_DIR, "configs/config_train_bert_hete_gcn.json")
config = load_config_json(CONFIG_PATH)

v = SimpleNamespace(**config)  # store v in config

word_embeddings_dim = 300
word_vector_map = {}  # this is not used, there is no word vector initialization


'''
    doc_name_list -> to store id info, which group it belongs and label
'''
doc_name_list, doc_train_list, doc_test_list = [], [], []

f = open('data/' + v.dataset + '.txt', 'r')
lines = f.readlines()

for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()

'''
    doc_content_list -> store raw texts in list, each element is data
                        instance
'''
doc_content_list = []

f = open('data/corpus/' + v.dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
print("Total number of documents: " + str(len(doc_content_list)))


'''
    Get train and test ids, randomly shuffle them and store them in
    new txt file.
'''

train_ids = [int(x.split('\t')[0]) for x in doc_train_list]
test_ids = [int(x.split('\t')[0]) for x in doc_test_list]

print("Number of initial training documents: " + str(len(train_ids)))
print("Number of initial test documents: " + str(len(test_ids)))

random.shuffle(train_ids)
random.shuffle(test_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + v.dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + v.dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids  # len(ids) = 10662

'''
    Obtain the shuffled version of the doc_name_list and doc_words_list
    lists using the shuffled ids. Write the string version of the lists
    to txt file.
'''

shuffle_doc_name_list, shuffle_doc_words_list = [], []

for id in ids:
    shuffle_doc_name_list.append(doc_name_list[id])
    shuffle_doc_words_list.append(doc_content_list[id])

shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + v.dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + v.dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

'''
    Build vocab
'''

word_freq = {}
word_set = set()
for docs in shuffle_doc_words_list:
    words = docs.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab, vocab_size = list(word_set), len(list(word_set))

print("Vocab size is: " + str(vocab_size))

'''
    Build dictionary where keys are unique words and value is the list of
    unique documents that this word appears
'''

word_doc_list = {}

for i, doc in enumerate(shuffle_doc_words_list):
    words = doc.split()
    appeared = set()
    for word in words:
        if word in appeared:
            # skip that iteration and go to next for loop element
            continue
        if word in word_doc_list:
            # create temp doc_list of existing document list
            tmp = word_doc_list[word]
            tmp.append(i)
            word_doc_list[word] = tmp
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

'''
    Build frequency dictionary for unique words, if word_x occurs in doc_y multiple times
    we only increment it by 1
'''
word_doc_freq = {}

for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}

for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)
f = open('data/corpus/' + v.dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
    label list
'''
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + v.dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

'''
    select 90% of training set, rest is for validation
'''

train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + v.dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()


'''
    Create input training matrix x -> (real_train_size,word_embeddings_dim)
    Create input label matrix y -> one hot version of labels

    Note: Initially it is empty since word_vector_map commented in original code,
          it might be useful for future implementations so we do not change it
'''
row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:  # never enters here
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))
print("\n***After train val split***")
print('Number of real training documents: ' + str(real_train_size))
y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)

# note that y is one hot version of the labels, if data has label 0 = [1 0], if 1 = [0 1] for binary case
y = np.array(y)
test_size = len(test_ids)

'''
    Create input matrix tx and label matrix ty for test set
'''

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        data_tx.append(doc_vec[j] / doc_len)

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)

'''
    Create feature vectors of both labeled and unlabeled training instances
    all_x -> (train_size + vocab_size, word_embedding), X in TextGCN
'''

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:  # buraya girmiyor başta
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)


'''
    We can define these matrices in the TextGCN notation:
    We split the X matrix into components but it is not one-hot anymore
'''

'''
    From now on, we will create our graphs
'''

'''
    Word-Word (PMI) graph
'''

window_size, windows = 20, []

# Create all windows
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    l = len(words)
    if l <= window_size:
        windows.append(words)
    else:
        for j in range(l - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

# W(i) in TextGCN = word_window_freq,
# number of sliding windows in a corpus that contains word i

word_window_freq = {}
for window in windows:
    appeared = set()
    for i, word in enumerate(window):
        if word in appeared:
            continue
        if word in word_window_freq:
            word_window_freq[word] += 1
        else:
            word_window_freq[word] = 1
        appeared.add(word)

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]

            if word_i_id == word_j_id:
                continue

            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

# PMI Matrix, it is symmetric since word_pair_count dict is symmetric

row, col, weight = [], [], []  # to have them in single graph
weight_tfidf, weight_pmi = [], []

row_nf, col_nf, weight_nf = [], [], []
'''
    We calculate PMI score with word_pair_count & word_window_freq
    note that adjacency matrix has documents first then words, so in original
    implementation row & col indices for sparse matrix incremented with trainsize
'''

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    # for big Adjacency
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)
    weight_pmi.append(pmi)
    weight_tfidf.append(1e-8)

# TF-IDF matrix
doc_word_freq = {}

'''
    calculating term frequency
'''
for doc_id, doc_words in enumerate(shuffle_doc_words_list):
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1


'''
    calculating inverse document frequency
'''
for i, doc_words in enumerate(shuffle_doc_words_list):
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        k = 0  # to track test documents
        train_flag, test_flag = False, False
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
            k = k + 1

        col.append(train_size + j)

        row_nf.append(i)
        col_nf.append(j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])

        weight.append(freq * idf)
        weight_tfidf.append(freq*idf)
        weight_nf.append(freq*idf)

        weight_pmi.append(1e-8)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


'''
    Creating sparse matrices from row,col and weights
'''

adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

adj_pmi = sp.csr_matrix(
    (weight_pmi, (row, col)), shape=(node_size, node_size))

adj_tfidf = sp.csr_matrix(
    (weight_tfidf, (row, col)), shape=(node_size, node_size))

adj_nf = sp.csr_matrix(
    (weight_nf, (row_nf, col_nf)), shape=(train_size+test_size, vocab_size))
'''
    After creating sparse matrices, we dump them to pickle objects
'''


def dump_obj(obj, obj_name, dataset):

    f = open("data/ind.{}.{}".format(dataset, obj_name), 'wb')
    pkl.dump(obj, f)
    f.close()


objs = [(x, 'x'), (y, 'y'), (tx, 'tx'), (ty, 'ty'), (allx, 'allx'), (ally, 'ally'), (adj, 'adj'),
        (adj_pmi, 'adj_pmi'), (adj_tfidf, 'adj_tfidf'), (adj_nf, 'adj_nf')]

for obj_tuple in objs:
    obj, obj_name = obj_tuple
    dump_obj(obj, obj_name, v.dataset)

### END OF BUILD GRAPH PROCEDURE ###

# %%
