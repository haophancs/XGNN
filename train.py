import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from configs import arg_parse
from models import *
from utils import *

adj_list, features_list, labels, idx_map, idx_train, idx_val, idx_test = load_split_MUTAG_data()

args = arg_parse()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Model and optimizer

# model = GCN(nfeat=features.shape[1],
#             nclass=labels.max().item() + 1,
#             dropout=args.dropout)

model = GCN(nfeat=features_list[0].shape[1],
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()

if args.cuda:
    model.cuda()
    features = features_list.cuda()
    adj = adj_list.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    # # Split
    output = None
    for i in idx_train:
        if output is None:
            output = model(features_list[i], adj_list[i], idx_map)
        else:
            output = torch.vstack((output, model(features_list[i], adj_list[i], idx_map)))
    loss_train = F.cross_entropy(output, labels[idx_train])
    acc_train = accuracy(output, labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = None
    for i in idx_val:
        if output is None:
            output = model(features_list[i], adj_list[i], idx_map)
        else:
            output = torch.vstack((output, model(features_list[i], adj_list[i], idx_map)))
    loss_val = F.cross_entropy(output, labels[idx_val])
    acc_val = accuracy(output, labels[idx_val])

    # # Not split
    # output = model(features, adj, idx_map)
    # TODO : Determine LOSS FUNCTION
    # loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    # loss_train.backward()
    # optimizer.step()

    # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_train, acc_train, loss_val, acc_val


class EarlyStopping():
    def __init__(self, patience=10, min_loss=0.5, hit_min_before_stopping=False):
        self.patience = patience
        self.counter = 0
        self.hit_min_before_stopping = hit_min_before_stopping
        if hit_min_before_stopping:
            self.min_loss = min_loss
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                if self.hit_min_before_stopping == True and loss > self.min_loss:
                    print("Cannot hit mean loss, will continue")
                    self.counter -= self.patience
                else:
                    self.early_stop = True
        else:
            self.best_loss = loss
            counter = 0


# Train model
t_total = time.time()
early_stopping = EarlyStopping(10, hit_min_before_stopping=True)

for epoch in range(10000):
    loss_train, acc_train, loss_val, acc_val = train(epoch)
    print(loss_val)
    early_stopping(loss_val)
    if early_stopping.early_stop == True:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

output = None
for i in idx_test:
    if output is None:
        output = model(features_list[i], adj_list[i], idx_map)
    else:
        output = torch.vstack((output, model(features_list[i], adj_list[i], idx_map)))
loss_test = F.cross_entropy(output, labels[idx_test])
acc_test = accuracy(output, labels[idx_test])
print(loss_test)
print(acc_test)

torch.save(model.state_dict(), args.outputdir)
