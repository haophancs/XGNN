import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graphconv import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        """ As per paper """
        """ 3 layers of GCNs with output dimensions equal to 32, 48, 64 respectively and average all node features """
        """ Final classifier with 2 fully connected layers and hidden dimension set to 32 """
        """ Activation function - ReLu (Mutag) """

        super(GCN, self).__init__()

        self.dropout = dropout

        self.gc1 = GraphConvolution(nfeat, 32)
        self.gc2 = GraphConvolution(32, 48)
        self.gc3 = GraphConvolution(48, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, nclass)

    def forward(self, x, adj, idx_map):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))

        # prev = 0
        # y = []
        # for idx in idx_map:
        #   y.append(torch.mean(x[prev:idx_map[idx]], 0))
        #   prev = idx_map[idx]
        # y = torch.stack(y, 0)

        y = torch.mean(x, 0)

        y = F.relu(self.fc1(y))
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.softmax(self.fc2(y), dim=0)

        return y
