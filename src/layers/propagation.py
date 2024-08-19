
import torch as th
from torch import nn
from dgl import function as fn
import os
import torch
import dgl
import torch.nn.functional as F
from tools.args import parse_args



args = parse_args()
device = args.device


class Propagation(nn.Module):
    def __init__(self, k, alpha, edge_drop=0.):
        super(Propagation, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat):
        graph = graph.local_var().to(device)
        norm = th.pow(graph.in_degrees().float().clamp(min=1e-12), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp).to(feat.device)
        feat_0 = feat
        for _ in range(self._k):
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.edata['w'] = self.edge_drop(th.ones(graph.number_of_edges(), 1).to(feat.device))
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            feat = feat * norm
            feat = (1 - self._alpha) * feat + self._alpha * feat_0

        return feat
