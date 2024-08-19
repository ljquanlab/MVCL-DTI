import torch
import torch.nn as nn
import math

class BasicAttention(nn.Module):
    def __init__(self, embed_dim):
        super(BasicAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_probs = self.softmax(attention_scores)
        
        attention_output = torch.matmul(attention_probs, V)
        return attention_output
