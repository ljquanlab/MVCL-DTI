import torch as th
import torch.nn as nn

from layers.DistMult import DistMult
from layers.ne_encoder import NeEncoder
from tools.tools import l2_norm
from layers.contrast import Contrast
from layers.mp_encoder import MpEncoder
from layers.df_encoder import DfEncoder
from tools.args import parse_args



args = parse_args()
device = args.device
drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'
go = 'go'
substituents = 'substituents'
drug_len = 708
protein_len = 1512
disease_len = 5603
sideeffect_len = 4192
go_len = 9279
substituents_len = 931
node_type_drug = th.zeros(708, dtype=th.long)  
node_type_protein = th.ones(1512, dtype=th.long)
node_type_disease = th.full((disease_len,), 2, dtype=th.long)  
node_type_go = th.full((go_len,), 3, dtype=th.long)  
node_type_substituents = th.full((substituents_len,), 4, dtype=th.long)  
node_type_sideeffect = th.full((sideeffect_len,), 5, dtype=th.long)  
node_type_drug = node_type_drug.to(device)
node_type_protein = node_type_protein.to(device)
node_type_disease = node_type_disease.to(device)
node_type_go = node_type_go.to(device)
node_type_substituents = node_type_substituents.to(device)
node_type_sideeffect = node_type_sideeffect.to(device)

node_type_mapping = {
    'drug': node_type_drug,
    'protein': node_type_protein,
    'disease': node_type_disease,
    'go': node_type_go,
    'substituents': node_type_substituents,
    'sideeffect': node_type_sideeffect
}


class MVCL(nn.Module):
    def __init__(self, hid_dim, args, keys, mps_len_dict: dict, attn_drop, feat_dim: dict, g):
        super(MVCL, self).__init__()
        self.device = th.device(args.device)
        self.dim_embedding = hid_dim
        self.keys = keys
        self.reg_lambda = args.reg_lambda

        self.fc_dict = nn.ModuleDict({k: nn.Linear(v, hid_dim) for k, v in feat_dim.items()})
        self.neencoder = NeEncoder(hid_dim, keys, g)
        self.neencoder2 = NeEncoder(hid_dim, keys, g)
        self.mpencoder = nn.ModuleDict({k: MpEncoder(v, hid_dim, attn_drop) for k, v in mps_len_dict.items()})
        self.mpencoder2 = nn.ModuleDict({k: MpEncoder(v, hid_dim, attn_drop) for k, v in mps_len_dict.items()})
        self.dfencoder = DfEncoder(hid_dim, g)
        self.dfencoder2 = DfEncoder(hid_dim, g)
        self.contrast = Contrast(hid_dim, args.tau, keys)
        self.distmult = DistMult(self.dim_embedding)
        self.reset_parameters()

    def reset_parameters(self):
        for m in MVCL.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                protein_sequence, protein_disease, protein_go, drug_protein, drug_protein_mask,
                mps_dict: dict, pos_dict: dict, cl, node_feature: dict):
        node_f = {k: self.fc_dict[k](node_feature[k]) for k, v in node_feature.items()}
        node_ne, node_mp, node_df = node_f, node_f, node_f
        node_ne = self.neencoder(drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                                 protein_sequence, protein_disease, protein_go, drug_protein, node_ne)
        node_ne = self.neencoder2(drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                                  protein_sequence, protein_disease, protein_go, drug_protein, node_ne)

        node_mp = {k: self.mpencoder[k](node_mp[k], mps_dict[k], node_type_mapping[k]) for k, v in mps_dict.items()}
        node_mp = {k: self.mpencoder2[k](node_mp[k], mps_dict[k], node_type_mapping[k]) for k, v in mps_dict.items()}

        node_df = self.dfencoder(drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                                 protein_sequence, protein_disease, protein_go, drug_protein, node_df)
        node_df = self.dfencoder2(drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                                 protein_sequence, protein_disease, protein_go, drug_protein, node_df)
    
        
        node_ne, node_mp, node_df = {k: l2_norm(v) for k, v in node_ne.items()}, {k: l2_norm(v) for k, v in node_mp.items()}, {k: l2_norm(v) for k, v in node_df.items()}
    
        cl_loss1 = self.contrast(node_ne, node_mp, pos_dict) 
        cl_loss2 = self.contrast(node_ne, node_df, pos_dict) 
        cl_loss3 = self.contrast(node_mp, node_df, pos_dict) 
        cl_loss = cl_loss1 + cl_loss2 + cl_loss3
        
        node_act = node_ne
        
        disease_vector = node_act[disease]
        drug_vector = node_act[drug]
        protein_vector = node_act[protein]
        sideeffect_vector = node_act[sideeffect]
        go_vector = node_act[go]
        substituents_vector = node_act[substituents]

    
        mloss, dti_re = self.distmult(drug_vector, disease_vector, sideeffect_vector, protein_vector, go_vector, substituents_vector,
                                      drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                                      protein_sequence, protein_disease, protein_go, drug_protein, drug_protein_mask)
        
        L2_loss = 0. 
        for name, param in MVCL.named_parameters(self):
            if 'bias' not in name:
                L2_loss = L2_loss + th.sum(param.pow(2))
        L2_loss = L2_loss * 0.5
        loss = mloss + self.reg_lambda * L2_loss + cl * cl_loss 
        return loss, dti_re.detach()
