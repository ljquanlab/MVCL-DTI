import dgl
import torch.nn.functional as F
import torch as th
import torch.nn as nn

from layers.SemanticsAttention import SemanticsAttention
from tools.tools import row_normalize
from layers.propagation import Propagation
from tools.args import parse_args
from tools.tools import l2_norm



args = parse_args()
import math

import os
import torch

drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'
go = 'go'
substituents = 'substituents'

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=0.3):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim // n_heads)

    def forward(self, X):
        bsz = X.shape[0]

        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
      
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        attention = torch.nn.Softmax(dim=-1)(energy)
        
        x = torch.matmul(attention, V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        
        x = self.fc(x)
        

        return x

class AttentionWeightedSum(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(AttentionWeightedSum, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, attention_dim)
        self.output_fc = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_scores = self.attention_fc(x)  
        attention_scores = F.relu(attention_scores)
        attention_scores = self.output_fc(attention_scores)  
        attention_scores = self.softmax(attention_scores)  

        weighted_sum = torch.sum(attention_scores * x, dim=1)  

        return weighted_sum, attention_scores.squeeze(-1)

class DfEncoder(nn.Module):
    def __init__(self, out_dim, g):
        super(DfEncoder, self).__init__()
        self.g = g
        self.dim_embedding = out_dim

        self.activation = F.elu
        self.reg_lambda = args.reg_lambda

        self.num_disease = 5603
        self.num_drug = 708
        self.num_protein = 1512
        self.num_sideeffect = 4192
        self.num_go = 9279
        self.num_substituents = 931

        
        self.fc_DDI = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_ch = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_Di = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_Side = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_Subs = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_PPI = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_seq = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_Di = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_GO = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Di_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Di_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Side_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Subs_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_GO_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()


        self.att_sum = AttentionWeightedSum(2048, 128)
        self.att_sum_1 = AttentionWeightedSum(2048, 128)
        self.att_sum_2 = AttentionWeightedSum(2048, 128)
        self.att_sum_3 = AttentionWeightedSum(2048, 128)
        self.att_sum_4 = AttentionWeightedSum(2048, 128)
        self.att_sum_5 = AttentionWeightedSum(2048, 128)


        self.propagation = Propagation(args.k, args.alpha, args.edge_drop)
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in DfEncoder.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                protein_sequence, protein_disease, protein_go, drug_protein, node_feature: dict)-> dict:

        disease_feat = node_feature[disease]
        drug_feat = node_feature[drug]
        protein_feat = node_feature[protein]
        sideeffect_feat = node_feature[sideeffect]
        go_feat = node_feature[go]
        substituents_feat = node_feature[substituents]
        disease_feat, att_s = self.att_sum(th.stack((th.mm(row_normalize(drug_disease.T).float(),
                                               F.relu(self.fc_Di_D(drug_feat))),
                                         th.mm(row_normalize(protein_disease.T).float(),
                                               F.relu(self.fc_Di_P(protein_feat))),
                                         disease_feat), dim=1))

        drug_feat, att_s1 = self.att_sum_1(th.stack((th.mm(row_normalize(drug_drug).float(),
                                            F.relu(self.fc_DDI(drug_feat))),
                                      th.mm(row_normalize(drug_chemical).float(),
                                            F.relu(self.fc_D_ch(drug_feat))),
                                      th.mm(row_normalize(drug_disease).float(),
                                            F.relu(self.fc_D_Di(disease_feat))),

                                      th.mm(row_normalize(drug_sideeffect).float(),
                                            F.relu(self.fc_D_Side(sideeffect_feat))),
                                      th.mm(row_normalize(drug_protein).float(),
                                            F.relu(self.fc_D_P(protein_feat))),
                                      th.mm(row_normalize(drug_substituents).float(),  F.relu(self.fc_D_Subs(substituents_feat))),
                                      drug_feat), dim=1))
        
        protein_feat, att_s2 = self.att_sum_2(th.stack((th.mm(row_normalize(protein_protein).float(),
                                               F.relu(self.fc_PPI(protein_feat))),
                                         th.mm(row_normalize(protein_sequence).float(),
                                               F.relu(self.fc_P_seq(protein_feat))),
                                         th.mm(row_normalize(protein_disease).float(),
                                               F.relu(self.fc_P_Di(disease_feat))),
                                         th.mm(row_normalize(drug_protein.T).float(),
                                               F.relu(self.fc_P_D(drug_feat))),
                                        th.mm(row_normalize(protein_go).float(), F.relu(self.fc_P_GO(go_feat))),
                                         protein_feat), dim=1))


        sideeffect_feat, att_s3 = self.att_sum_3(th.stack((th.mm(row_normalize(drug_sideeffect.T).float(),
                                                  F.relu(self.fc_Side_D(drug_feat))),
                                            sideeffect_feat), dim=1))
        
        substituents_feat, att_s4 = self.att_sum_4(th.stack((th.mm(row_normalize(drug_substituents.T).float(),
                                                  F.relu(self.fc_Subs_D(drug_feat))),
                                            substituents_feat), dim=1))
        
        go_feat, att_s5 = self.att_sum_5(th.stack((th.mm(row_normalize(protein_go.T).float(),
                                                  F.relu(self.fc_GO_P(protein_feat))),
                                            go_feat), dim=1))


        node_feat = th.cat((disease_feat, drug_feat, protein_feat, sideeffect_feat, go_feat, substituents_feat), dim=0)

        node_feat = self.propagation(dgl.to_homogeneous(self.g), node_feat)


        disease_embedding = node_feat[:self.num_disease]
        drug_embedding = node_feat[self.num_disease:self.num_disease + self.num_drug]
        protein_embedding = node_feat[self.num_disease + self.num_drug:self.num_disease + self.num_drug +
                                                                       self.num_protein]
        sideeffect_embedding = node_feat[self.num_disease+self.num_drug+self.num_protein:self.num_disease+self.num_drug+self.num_protein+self.num_sideeffect]
        go_embedding = node_feat[self.num_disease+self.num_drug+self.num_protein+self.num_sideeffect:self.num_disease+self.num_drug+self.num_protein+self.num_sideeffect+self.num_go]
        substituents_embedding = node_feat[-self.num_substituents:]
        disease_vector = l2_norm(disease_embedding)
        drug_vector = l2_norm(drug_embedding)
        protein_vector = l2_norm(protein_embedding)
        sideeffect_vector = l2_norm(sideeffect_embedding)
        go_vector = l2_norm(go_embedding)
        substituents_vector = l2_norm(substituents_embedding)


        return {drug: drug_vector, protein: protein_vector, sideeffect: sideeffect_vector, disease: disease_vector, go:go_vector, substituents:substituents_vector}