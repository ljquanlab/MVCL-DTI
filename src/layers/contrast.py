import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Attention Module for Multi-view Integration
class Attention(nn.Module):
    def __init__(self, out_dim, num_heads, attn_vec_dim):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(out_dim, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        # Weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, view_outs):
        beta = []
        for view_out in view_outs:
            fc1 = torch.tanh(self.fc1(view_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        view_outs = [torch.unsqueeze(view_out, dim=0) for view_out in view_outs]
        view_outs = torch.cat(view_outs, dim=0)
        h = torch.sum(beta * view_outs, dim=0)
        return h

class Contrast(nn.Module): 
    def __init__(self, out_dim, tau, keys): 
        super(Contrast, self).__init__()
        self.attention = Attention(out_dim, num_heads=4, attn_vec_dim=64)  # 使用 Attention 模块
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
        z_mp_att = self.attention([z_mp])
        z_ne_att = self.attention([z_ne])
        
        # Combine original and attention-based features for contrastive learning
        z_mp_combined = z_mp + z_mp_att
        z_ne_combined = z_ne + z_ne_att
    
        return self.compute_loss(z_mp_combined, z_ne_combined, pos, k)


    def forward(self, z_mp, z_ne, pos):
        sumLoss = 0
        for k, v in pos.items(): 
            sumLoss += self.compute_loss_with_attention(z_mp[k], z_ne[k], pos[k], k) 
        return sumLoss
