import torch.nn.functional as F
import torch as th
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import dgl

from layers.SemanticsAttention import SemanticsAttention
from tools.tools import row_normalize
from layers.propagation import Propagation
from tools.args import parse_args
from tools.tools import l2_norm




args = parse_args()


drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'
go = 'go'
substituents = 'substituents'

class NeEncoder(nn.Module):
    def __init__(self, out_dim, keys, g):
        super(NeEncoder, self).__init__()
        self.dim_embedding = out_dim
        self.keys = keys

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
    
        self.reset_parameters()

    
    def reset_parameters(self):
        for m in NeEncoder.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                protein_sequence, protein_disease, protein_go, drug_protein, node_feature: dict) -> dict:
        disease_feature = node_feature[disease]
        drug_feature = node_feature[drug]
        protein_feature = node_feature[protein]
        sideeffect_feature = node_feature[sideeffect]
        go_feature = node_feature[go]
        substituents_feature = node_feature[substituents]
       
        disease_agg = [th.mm(row_normalize(drug_disease.T).float(), F.relu(self.fc_Di_D(drug_feature))),
                       th.mm(row_normalize(protein_disease.T).float(), F.relu(self.fc_Di_P(protein_feature))),
                       disease_feature]
        drug_agg = [th.mm(row_normalize(drug_drug).float(), F.relu(self.fc_DDI(drug_feature))),
                    th.mm(row_normalize(drug_chemical).float(), F.relu(self.fc_D_ch(drug_feature))),
                    th.mm(row_normalize(drug_disease).float(), F.relu(self.fc_D_Di(disease_feature))),
                    th.mm(row_normalize(drug_sideeffect).float(), F.relu(self.fc_D_Side(sideeffect_feature))),
                    th.mm(row_normalize(drug_protein).float(), F.relu(self.fc_D_P(protein_feature))),
                    th.mm(row_normalize(drug_substituents).float(), F.relu(self.fc_D_Subs(substituents_feature))),
                    drug_feature]
        protein_agg = [th.mm(row_normalize(protein_protein).float(), F.relu(self.fc_PPI(protein_feature))),
                       th.mm(row_normalize(protein_sequence).float(), F.relu(self.fc_P_seq(protein_feature))),
                       th.mm(row_normalize(protein_disease).float(), F.relu(self.fc_P_Di(disease_feature))),
                       th.mm(row_normalize(protein_go).float(), F.relu(self.fc_P_GO(go_feature))),
                       th.mm(row_normalize(drug_protein.T).float(), F.relu(self.fc_P_D(drug_feature))),
                       protein_feature]
        sideeffect_agg = [th.mm(row_normalize(drug_sideeffect.T).float(), F.relu(self.fc_Side_D(drug_feature))),
                          sideeffect_feature]
        substituents_agg = [th.mm(row_normalize(drug_substituents.T).float(), F.relu(self.fc_Subs_D(drug_feature))),
                          substituents_feature]
        go_agg = [th.mm(row_normalize(protein_go.T).float(), F.relu(self.fc_GO_P(protein_feature))),
                          go_feature]
    
        disease_feat = th.mean(th.stack(disease_agg, dim=1), dim=1)
        drug_feat = th.mean(th.stack(drug_agg, dim=1), dim=1)
        protein_feat = th.mean(th.stack(protein_agg, dim=1), dim=1)
        sideeffect_feat = th.mean(th.stack(sideeffect_agg, dim=1), dim=1)
        substituents_feat = th.mean(th.stack(substituents_agg, dim=1), dim=1)
        go_feat = th.mean(th.stack(go_agg, dim=1), dim=1)

        return {drug: drug_feat, protein: protein_feat, sideeffect: sideeffect_feat, disease: disease_feat, substituents: substituents_feat, go: go_feat}
