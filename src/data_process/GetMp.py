import numpy as np
import scipy.sparse as sp

from tools.tools import sparse_mx_to_torch_sparse_tensor, normalize_adj


def get_mp(drug_drug, durg_protein, drug_disease, drug_sideeffect, drug_substituents, protein_protein, protein_disease, protein_go, device):
    drdr = drug_drug
    drpr = durg_protein
    prpr = protein_protein
    drdi = drug_disease
    prdi = protein_disease
    drse = drug_sideeffect
    drsu = drug_substituents
    prdi = protein_disease
    prgo = protein_go

    drprdr = np.matmul(drpr, drpr.T) > 0
    prdrpr = np.matmul(drpr.T, drpr) > 0
    drprpr = np.matmul(drpr, prpr) > 0
    drprprdr = np.matmul(drprpr, drpr.T) > 0
    drprdr = np.array(drprdr)
    prdrpr = np.array(prdrpr)
    prdrdr = np.matmul(drpr.T, drdr) > 0
    prdrdrpr = np.matmul(prdrdr, drpr) > 0
    drdidr = np.matmul(drdi, drdi.T) > 0
    drdidr = np.array(drdidr)
    prdipr = np.matmul(prdi, prdi.T) > 0
    prdipr = np.array(prdipr)
    drsedr = np.matmul(drse, drse.T) > 0
    drsedr = np.array(drsedr)
    drsudr = np.matmul(drsu, drsu.T) > 0
    drsudr = np.array(drsudr)
    prgopr = np.matmul(prgo, prgo.T) > 0
    prgopr = np.array(prgopr)
    


    drdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drdr))).to(device)
    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdr))).to(device)
    drprprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprprdr))).to(device)
    drdidr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drdidr))).to(device)
    drsedr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drsedr))).to(device)
    drsudr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drsudr))).to(device)

    prpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prpr))).to(device)
    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrpr))).to(device)
    prdrdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrdrpr))).to(device)
    prdipr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdipr))).to(device)
    prgopr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prgopr))).to(device)

    drug = 'drug'
    protein = 'protein'
    disease = 'disease'
    sideeffect = 'sideeffect'
    go = 'go'
    substituents = 'substituents'

    mps_dict = {drug: [drdr, drprdr, drdidr, drsedr, drsudr], protein: [prpr, prdrpr, prdipr, prgopr], disease: [], sideeffect: [], go: [], substituents: []}

    return mps_dict
