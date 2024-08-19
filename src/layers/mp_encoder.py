import time

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling


sys.path.append('../')
from layers.SemanticsAttention import SemanticsAttention

class GNNFiLM(nn.Module):
    def __init__(self, in_ft, out_ft, num_types, bias=True): 
        super(GNNFiLM, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False) 
        self.act = nn.PReLU() 
        
        self.fc_gamma = nn.Linear(num_types, out_ft)  
        self.fc_beta = nn.Linear(num_types, out_ft)   

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, node_type):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        node_type_one_hot = F.one_hot(node_type, num_classes=6).float()  
        gamma = self.fc_gamma(node_type_one_hot)
        beta = self.fc_beta(node_type_one_hot)
        out = (gamma * out) + beta
        if self.bias is not None:
            out += self.bias
        return self.act(out)



class MpEncoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop, num_types=6): 
        super(MpEncoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GNNFiLM(hidden_dim, hidden_dim, num_types) for _ in range(P)])
        self.att = SemanticsAttention(hidden_dim, attn_drop)

    def forward(self, h, mps, node_type):
        if self.P == 0:
            return h
        embeds = []
        for i, mp in enumerate(mps):  
            embeds.append(self.node_level[i](h, mp, node_type))  
        z_mp = self.att(embeds) 
        return z_mp
