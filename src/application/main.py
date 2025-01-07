# -*- coding: utf-8 -*-
import random
import time
import os

import torch
import torch as th
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
import gc
import joblib
import pickle

sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_process.GetMp import get_mp
from data_process.GetPos import get_pos
from model.MVCL import MVCL
from tools.args import parse_args
from tools.tools import load_data, ConstructGraph, load_feature, compute_auc_aupr
import warnings
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

warnings.filterwarnings('ignore')

DR_DR_V = 'drug_drug virtual'
PR_PR_V = 'protein_protein virtual'
DI_DI_V = 'disease_disease virtual'
SE_SE_V = 'sideeffect_sideeffect virtual'
G_G_V = 'go_go virtual'
SU_SU_V = 'substituents_substituents virtual'

DR_PR_I = 'drug_protein interaction'
PR_DR_I = 'protein_drug interaction'

DR_DR_A = 'drug_drug association'
DR_PR_A = 'drug_protein association'
DR_DI_A = 'drug_disease association'
DR_SE_A = 'drug_sideeffect association'
DR_G_A = 'drug_go association'
DR_SU_A = 'drug_substituents association'

PR_DR_A = 'protein_drug association'
PR_PR_A = 'protein_protein association'
PR_DI_A = 'protein_disease association'
PR_SE_A = 'protein_sideeffect association'
PR_G_A = 'protein_go association'
PR_SU_A = 'protein_substituents association'

DI_DR_A = 'disease_drug association'
DI_PR_A = 'disease_protein association'
DI_DI_A = 'disease_disease association'
DI_SE_A = 'disease_sideeffect association'
DI_G_A = 'disease_go association'
DI_SU_A = 'disease_substituents association'

SE_DR_A = 'sideeffect_drug association'
SE_PR_A = 'sideeffect_protein association'
SE_DI_A = 'sideeffect_disease association'
SE_SE_A = 'sideeffect_sideeffect association'
SE_G_A = 'sideeffect_go association'
SE_SU_A = 'sideeffect_substituents association'

G_DR_A = 'go_drug association'
G_PR_A = 'go_protein association'
G_DI_A = 'go_disease association'
G_SE_A = 'go_sideeffect association'
G_G_A = 'go_go association'
G_S_A = 'go_substituents association'

SU_DR_A = 'substituents_drug association'
SU_PR_A = 'substituents_protein association'
SU_DI_A = 'substituents_disease association'
SU_SE_A = 'substituents_sideeffect association'
SU_G_A = 'substituents_go association'
SU_SU_A = 'substituents_substituents association'

drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'
go = 'go'
substituents = 'substituents'
relation_dti = ('drug', 'drug_protein interaction', 'protein')

drug_len = 708
protein_len = 1512
sideeffect_len = 4192
disease_len = 5603
go_len = 9279
su_len = 931

args = parse_args()
print(args)
device = args.device
seeds = [538, 456, 789]
DRDRDR, DRPRDR, DRSEDR, DRDIDR, DRSUDR = 'DRDRDR', 'DRPRDR', 'DRSEDR', 'DRDIDR', 'DRSUDR'
PRDRPR, PRPRPR, PRDIPR, PRGPR = 'PRDRPR', 'PRPRPR', 'PRDIPR', 'PRGPR'
SEDRSE = 'SEDRSE'
DIDRDI, DIPRDI = 'DIDRDI', "DIPRDI"
GPRG = 'GPRG'
SUDRSU = 'SUDRSU'


def TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_drug, drug_chemical, drug_disease,
                     drug_sideeffect, drug_substituents, protein_protein, protein_sequence, protein_disease, protein_go):
    best_valid_aupr, test_aupr, test_auc, patience = 0., 0., 0., 0.
    train_label = th.tensor(DTItrain[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    val_label = th.tensor(DTIvalid[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    test_label = th.tensor(DTItest[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    drug_protein = th.zeros((drug_len, protein_len))
    mask = th.zeros((drug_len, protein_len)).to(device) 
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]], mask[ele[0], ele[1]] = ele[2], 1
    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, drug_substituents, protein_protein,
                                  protein_sequence, protein_disease, protein_go, drug_protein)
    dti_np = drug_protein.numpy()
    pos_dict = get_pos(hetero_graph, device)
    mps_dict = get_mp(drug_drug, dti_np, drug_disease, drug_sideeffect, drug_substituents, protein_protein, protein_disease, protein_go, device)

    mp_len_dict = {k: len(v) for k, v in mps_dict.items()}
    drug_dr = th.tensor(drug_drug).to(device)
    drug_ch = th.tensor(drug_chemical).to(device)
    drug_di = th.tensor(drug_disease).to(device)
    drug_se = th.tensor(drug_sideeffect).to(device)
    drug_su = th.tensor(drug_substituents).to(device)
    protein_pr = th.tensor(protein_protein).to(device)
    protein_seq = th.tensor(protein_sequence).to(device)
    protein_di = th.tensor(protein_disease).to(device)
    protein_g = th.tensor(protein_go).to(device)
    drug_pr = drug_protein.to(device)

    node_feature, feat_dim = load_feature(args.feature)
    keys = [drug, protein, disease, sideeffect, go, substituents]
    model = MVCL(args.hid_dim, args, keys, mp_len_dict, args.attn_drop, feat_dim, hetero_graph).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.epochs):
        model.train()
        loss, dp_re = model(drug_dr, drug_ch, drug_di, drug_se, drug_su, protein_pr, protein_seq, protein_di, protein_g, drug_pr,
                            mask, mps_dict, pos_dict, args.cl, node_feature)
        results = dp_re.detach()
        train_pre = results[DTItrain[:, 0], DTItrain[:, 1]]
        train_auc, train_aupr = compute_auc_aupr(train_pre, train_label)
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1) 
        optimizer.step()
        model.eval()
        with th.no_grad(): 
            val_pre = results[DTIvalid[:, 0], DTIvalid[:, 1]]
            valid_auc, valid_aupr = compute_auc_aupr(val_pre, val_label)
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience = 0
                test_pre = results[DTItest[:, 0], DTItest[:, 1]]
                test_auc, test_aupr = compute_auc_aupr(test_pre, test_label) 
            else: 
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break
        if i % 200 == 0:
            print(
                "Epoch {:05d} | Train Loss {:02f} | Train auc {:.4f} | Train aupr {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f} | Test ROC_AUC {:.4f} | Test AUPR {:.4f}"
                    .format(i, loss.item(), train_auc, train_aupr, valid_auc, valid_aupr, test_auc, test_aupr))
    return test_auc, test_aupr


def main(random_seed, task_name, dti_path='mat_drug_protein.txt'):
    drug_d, drug_ch, drug_di, drug_side, drug_su, protein_p, protein_seq, protein_di, protein_g, dti_original = load_data(dti_path)
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_original[i][j]) == 0:
                whole_negative_index.append([i, j])
    if args.number == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=10 * len(whole_positive_index), replace=False)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    elif args.number == 'one':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    else: 
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=len(whole_negative_index), replace=False)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)

    count = 0
    for i in whole_positive_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = i[0], i[1], 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = \
            whole_negative_index[i][0], whole_negative_index[i][1], 0
        count += 1

    print("----------------------------------------")
    print('random_seed=', str(random_seed), 'task=', task_name)
    print("----------------------------------------")
    test_auc_fold = []
    test_aupr_fold = []
    kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True) 
    for fold_index, (train_index, test_index) in enumerate(kf.split(data_set[:, :2], data_set[:, 2])):
        train, DTItest = data_set[train_index], data_set[test_index]
        DTItrain, DTIvalid = train_test_split(train, test_size=0.1, random_state=0)
        print("--------------------------------------------------------------")
        print("KFold ", str(fold_index), " of 10")
        print("--------------------------------------------------------------")
        time_roundStart = time.time()
        t_auc, t_aupr = TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_d, drug_ch, drug_di, drug_side, drug_su,
                                         protein_p, protein_seq, protein_di, protein_g)
        test_auc_fold.append(t_auc)
        test_aupr_fold.append(t_aupr)
        time_roundEnd = time.time()
        print('auc:', t_auc, 't_aupr', t_aupr)
        print("Time spent in this fold:", time_roundEnd - time_roundStart)
    test_auc_mean, test_aupr_mean = np.mean(test_auc_fold), np.mean(test_aupr_fold)
    print('mean_auc:', test_auc_mean, 'mean_aupr:', test_aupr_mean)
    return test_auc_mean, test_aupr_mean 

def main_unique(random_seed, task_name='unique'):
    drug_d, drug_ch, drug_di, drug_side, drug_su, protein_p, protein_seq, protein_di, protein_g, dti_original = \
        load_data(dti_path='mat_drug_protein_unique.txt')
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_original[i][j]) == 0:
                whole_negative_index.append([i, j])
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=10 * len(whole_positive_index), replace=False)
    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = i[0], i[1], 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = \
            whole_negative_index[i][0], whole_negative_index[i][1], 0
        count += 1
    whole_positive_index_test = []
    whole_negative_index_test = []
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 3:
                whole_positive_index_test.append([i, j])
            elif int(dti_original[i][j] == 2):
                whole_negative_index_test.append([i, j])
    negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),
                                                  size=10 * len(whole_positive_index_test), replace=False)
    data_set_test = np.zeros((len(negative_sample_index_test) + len(whole_positive_index_test), 3), dtype=int)
    count = 0
    for i in whole_positive_index_test:
        data_set_test[count][0], data_set_test[count][1], data_set_test[count][2] = i[0], i[1], 1
        count += 1
    for i in negative_sample_index_test:
        data_set_test[count][0], data_set_test[count][1], data_set_test[count][2] = \
            whole_negative_index_test[i][0], whole_negative_index_test[i][1], 0
        count += 1

    print("----------------------------------------")
    print('random_seed=', str(random_seed), 'task=', task_name)
    print("----------------------------------------")
    train_all, DTItest = data_set, data_set_test
    DTItrain, DTIvalid = train_test_split(train_all, test_size=0.1, random_state=0)
    t_auc, t_aupr = TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_d, drug_ch, drug_di, drug_side, drug_su,
                                     protein_p, protein_seq, protein_di, protein_g)
    return t_auc, t_aupr


def setup_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    task = args.task
    task = 'z5'
    description = 'cl=20000 mp=drdr drprdr drprprdr'
    file_name = ('' if task == 'benchmark' else '_' + task)
    file_name = ''

    number = args.number
    edge_mask = args.edge_mask
    print("----------------------------------------")
    print('task=', task, ' number=', number, ' filename=', file_name)
    print("----------------------------------------")
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(now_time)
    task_file_name = task + number + edge_mask
    print(seeds)
    print(description)

    folder_path = '../../result/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    time_main_start = time.time()
    with open(os.path.join(folder_path, task_file_name + '_auc'), 'a') as f:
        f.write(now_time + '\n')
        f.write(str(args) + '\n')
        f.write(str(seeds) + '\n')
        f.write(description + '\n')
    with open('../../result/' + task_file_name + '_aupr', 'a') as f:
        f.write(now_time + '\n')
        f.write(str(args) + '\n')
        f.write(str(seeds) + '\n')
        f.write(description + '\n')
    for select_seed in seeds:
        start = time.time()
        setup_seed(select_seed)
        if task == 'unique':
            auc, aupr = main_unique(select_seed)
        elif task == 'feature':
            auc, aupr = main(select_seed, task, dti_path='mat_drug_protein' + file_name + '.txt')
        else:
            auc, aupr = main(select_seed, task, dti_path='mat_drug_protein' + file_name + '.txt')
        with open('../../result/' + task_file_name + '_auc', 'a') as f:
            f.write(str(auc) + '\n')
        with open('../../result/' + task_file_name + '_aupr', 'a') as f:
            f.write(str(aupr) + '\n')
        end = time.time()
        print("Total time:", end - start)
    time_main_end = time.time()
    with open('../../result/' + task_file_name + '_auc', 'a') as f:
        f.write(str(time_main_end - time_main_start) + '\n')
    with open('../../result/' + task_file_name + '_aupr', 'a') as f:
        f.write(str(time_main_end - time_main_start) + '\n')
