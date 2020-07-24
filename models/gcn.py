import torch
import torch.nn as nn

import math


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x_ = nn.functional.relu(self.gc1(x, adj))
        x = nn.functional.dropout(x_, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x, x_

# class FC(nn.Module):
#     def __init__(self, in_features, nlabel, dropout=0.0):
#         super(FC, self).__init__()
#         self.dropout = dropout
#         self.fc1 = nn.Linear(in_features, nlabel, bias=True)

#         # self.fc1 = nn.Linear(in_features, 256, bias=True)
#         # self.fc2 = nn.Linear(256, 256, bias=True)
#         # self.fc3 = nn.Linear(256, 256, bias=True)
#         # self.fc4 = nn.Linear(256, 512, bias=True)
#         # self.fc5 = nn.Linear(512, 512, bias=True)
#         # self.fc6 = nn.Linear(512, nlabel, bias=True)

#     def forward(self, x):
#         # x = torch.tanh(self.fc1(x))
#         # x = torch.tanh(self.fc2(x))
#         # x = torch.tanh(self.fc3(x))
#         # x = torch.tanh(self.fc4(x))
#         # x = torch.tanh(self.fc5(x))
#         x = nn.functional.dropout(x, self.dropout, training=self.training)
#         return self.fc1(x)
