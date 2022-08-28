# %%
import torch as th
import torch.nn.functional as F
from model.models import BertClassifier, GCN_scratch
from utils import *
import torch.utils.data as Data
import os
import shutil
from model import BertGCN_sparse
import pandas as pd
import time

from types import SimpleNamespace
from pathlib import Path


WORK_DIR = Path(__file__).parent
CONFIG_PATH = Path.joinpath(
    WORK_DIR, "configs/config_train_bert_hete_gcn.json")
config = load_config_json(CONFIG_PATH)

v = SimpleNamespace(**config)  # store v in config

if v.checkpoint_dir == "":
    ckpt_dir = './checkpoint/{}_{}_{}'.format(
        v.bert_init, v.gcn_model, v.dataset)
else:
    ckpt_dir = v.checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)


cpu = th.device('cpu')
gpu = th.device('cuda:0')

_, _, _, adj_nf, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    v.dataset)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# model = BertGCN_sparse(nfeat=768, nb_class=nb_class, pretrained_model=v.bert_init, m=v.m,
#                        n_hidden=v.n_hidden, dropout=v.dropout)

'''
    trying seperate implementation
'''
model_bert = BertClassifier(pretrained_model=v.bert_init, nb_class=nb_class)


# if v.pretrained_bert_ckpt != "":
#     ckpt = th.load(v.pretrained_bert_ckpt, map_location=gpu)
#     model.bert_model.load_state_dict(ckpt['bert_model'])
#     model.classifier.load_state_dict(ckpt['classifier'])


input_ids, attention_mask = get_inputids_attention_mask(
    v, model_bert.tokenizer)

input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word,
                   v.max_length), dtype=th.long), input_ids[-nb_test:]])

attention_mask = th.cat([attention_mask[:-nb_test], th.zeros(
    (nb_word, v.max_length), dtype=th.long), attention_mask[-nb_test:]])

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)  # has shape (nb_node,)
doc_mask = train_mask + val_mask + test_mask

#data = pd.read_pickle(os.path.join('mr.pkl'))

'''
    here trying getting it from BertGCN
'''
NF = adj_nf
FN = adj_nf.T
NF = normalize_sparse_graph(NF, -0.5, -0.5)
FN = normalize_sparse_graph(FN, -0.5, -0.5)
NF = to_torch_sparse_tensor(NF)
FN = to_torch_sparse_tensor(FN)


idx_train, idx_val, idx_test = th.LongTensor(th.arange(0, nb_train)), th.LongTensor(th.arange(
    nb_train, nb_train+nb_val)), th.LongTensor(th.arange(nb_train+nb_val, nb_train+nb_val+nb_test))
labels = th.LongTensor(y[doc_mask])
labels = labels.cuda()

dataloader = Data.DataLoader(
    Data.TensorDataset(input_ids[doc_mask],
                       attention_mask[doc_mask]),
    batch_size=1024
)
full_features = get_bert_output(dataloader, model_bert, gpu)

# %%

'''
    trying implementation for BertGCN_scratch
'''
model_gcn = GCN_scratch(nfeat=768, n_hidden=v.n_hidden,
                        nclass=nb_class, dropout=v.dropout)

optimizer_bert = th.optim.Adam(model_bert.parameters(),
                               lr=v.bert_lr)

optimizer_gcn = th.optim.Adam(model_gcn.parameters(),
                              lr=v.gcn_lr, weight_decay=v.weight_decay)
model_gcn.cuda()
model_bert.cuda()

NF, FN = NF.cuda(), FN.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
full_features = full_features.cuda()

idx_train_dataset = Data.TensorDataset(idx_train)
idx_loader_train = Data.DataLoader(
    idx_train_dataset, v.batch_size, shuffle=True)

# %%

dataloader = Data.DataLoader(
    Data.TensorDataset(input_ids[doc_mask],
                       attention_mask[doc_mask]),
    batch_size=1024
)


def update_feature():
    global full_features, dataloader
    full_features = get_bert_output(dataloader, model_bert, gpu)
    full_features = full_features.cuda()


def mini_batch_train(epoch, idx_loader_train):
    t = time.time()
    print("Epoch: " + str(epoch))
    for batch_no, batch in enumerate(idx_loader_train):
        (idx, ) = [x for x in batch]
        train(batch_no, idx, epoch)


def train(batch_no, idx, epoch):
    if epoch >= 10:
        model_gcn.train()
        optimizer_gcn.zero_grad()
        gcn_pred = model_gcn(full_features, NF, FN, idx)
        loss_gcn = F.nll_loss(th.log(gcn_pred+1e-10), labels[idx])
        a, m1, m2 = get_metrics(th.log(gcn_pred)+1e-10, labels[idx])
        loss_gcn.backward()
        optimizer_gcn.step()

    model_bert.train()
    optimizer_bert.zero_grad()
    bert_pred = model_bert(input_ids.cuda(), attention_mask.cuda(), idx)
    loss_bert = F.nll_loss(th.log(bert_pred+1e-10), labels[idx])
    acc_train, macro_train, micro_train = get_metrics(th.log(bert_pred+1e-10),
                                                      labels[idx])
    loss_bert.backward()
    optimizer_bert.step()
    update_feature()

    if batch_no % 3 == 0:
        print('BERT results: ',
              'Batch: {:04d}'.format(batch_no+1),
              # 'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'macro_f1_train: {:.4f}'.format(macro_train.item()),
              'micro_f1_train: {:.4f}'.format(micro_train.item()))
        if epoch >= 10:
            print('GCN results: ',
                  'Batch: {:04d}'.format(batch_no+1),
                  # 'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(a.item()),
                  'macro_f1_train: {:.4f}'.format(m1.item()),
                  'micro_f1_train: {:.4f}'.format(m2.item()))


# def test():
#     model.eval()
#     output = model(full_features, NF, FN, input_ids.cuda(),
#                    attention_mask.cuda(), th.from_numpy(doc_mask).cuda(), idx_test)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test, macro_test, micro_test = get_metrics(output[idx_test],
#                                                    labels[idx_test])
#     print("Test set results:",
#           "accuracy= {:.4f}".format(acc_test.item()),
#           "macro_f1= {:.4f}".format(macro_test.item()),
#           "micro_f1= {:.4f}".format(micro_test.item()))


# Train model
t_total = time.time()
for epoch in range(v.nb_epochs):
    mini_batch_train(epoch, idx_loader_train)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
