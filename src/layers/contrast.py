import torch
import torch.nn as nn
from layers.attention import BasicAttention
import math


class Contrast(nn.Module): 
    def __init__(self, out_dim, tau, keys): 
        super(Contrast, self).__init__()
        self.attention = BasicAttention(out_dim)
        self.proj = nn.ModuleDict({k: nn.Sequential( 
            nn.Linear(out_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim)
        ) for k in keys})
        self.tau = tau
        for k, v in self.proj.items():
            for model in v:
                if isinstance(model, nn.Linear):
                    nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t()) 
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) 
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau) 
        return sim_matrix
        

    def compute_loss(self, z_mp, z_ne, pos, k): 
        z_proj_mp = self.proj[k](z_mp)   
        z_proj_ne = self.proj[k](z_ne)
        
        matrix_mp2ne = self.sim(z_proj_mp, z_proj_ne)
        matrix_ne2mp = matrix_mp2ne.t()
        
        softmax_mp2ne = torch.nn.functional.softmax(matrix_mp2ne, dim=1)
        lori_mp = -torch.log(softmax_mp2ne.mul(pos.to_dense()).sum(dim=-1)).mean() 

        softmax_ne2mp = torch.nn.functional.softmax(matrix_ne2mp, dim=1)
        lori_ne = -torch.log(softmax_ne2mp.mul(pos.to_dense()).sum(dim=-1)).mean() 

        return lori_mp + lori_ne 
    
    def compute_loss_with_attention(self, z_mp, z_ne, pos, k):
        z_mp_att = self.attention(z_mp, z_mp, z_mp)
        z_ne_att = self.attention(z_ne, z_ne, z_ne)
    
        return self.compute_loss(z_mp_att, z_ne_att, pos, k)


    def forward(self, z_mp, z_ne, pos):
        sumLoss = 0
        for k, v in pos.items(): 
            sumLoss += self.compute_loss_with_attention(z_mp[k], z_ne[k], pos[k], k) 
        return sumLoss
