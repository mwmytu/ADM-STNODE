# -*- coding: utf-8 -*-
# @Time : 2022/9/3 | 18:44
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : main.py
# Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from uitls import load_data, re_normalization, relations_to_matrix
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_train(CDA_train, circRNA_num, disease_num):
    CD_asso_train = np.zeros((circRNA_num, disease_num))
    # CD_asso_train_mask = np.zeros((circRNA_num, disease_num))
    pos_x_index = []
    pos_y_index = []
    neg_x_index = []
    neg_y_index = []
    for ele in CDA_train:
        CD_asso_train[ele[0], ele[1]] = ele[2]
        # CD_asso_train_mask[ele[0], ele[1]] = 1
        if ele[2] == 1:
            pos_x_index.append(ele[0])
            pos_y_index.append(ele[1])
        elif ele[2] == 0:
            neg_x_index.append(ele[0])
            neg_y_index.append(ele[1])
    return CD_asso_train, pos_x_index, pos_y_index, neg_x_index, neg_y_index


def evaluate(model, CD_asso_train, CDAvalid, CDAtest):
    model.eval()
    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0
    pred_list = []
    ground_truth = []
    with torch.no_grad():
        _, _, prediction_score = model(CD_asso_train)

    prediction_score = prediction_score.numpy()
    CDAvalid = CDAvalid.numpy()
    CDAtest = CDAtest.numpy()

    for ele in CDAvalid:
        pred_list.append(prediction_score[ele[0], ele[1]])
        ground_truth.append(ele[2])

    valid_auc = roc_auc_score(ground_truth, pred_list)
    # print (valid_auc)
    valid_aupr = average_precision_score(ground_truth, pred_list)

    if valid_aupr >= best_valid_aupr:

        best_valid_aupr = valid_aupr
        best_valid_auc = valid_auc
        pred_list = []
        ground_truth = []
        for ele in CDAtest:
            pred_list.append(prediction_score[ele[0], ele[1]])
            ground_truth.append(ele[2])
        test_auc = roc_auc_score(ground_truth, pred_list)
        fpr, tpr, thresholds = roc_curve(ground_truth, pred_list)
        test_aupr = average_precision_score(ground_truth, pred_list)
        precision, recall, thresholds = precision_recall_curve(ground_truth, pred_list)
    # print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr, fpr, tpr, precision, recall


def train_and_evaluate(CD_asso_train, CDAvalid, CDAtest, pos_x_index, pos_y_index, neg_x_index,
                       neg_y_index, circRNA_num, disease_num, args):
    pos_x_index = torch.tensor(pos_x_index, dtype=torch.long)
    pos_y_index = torch.tensor(pos_y_index, dtype=torch.long)
    neg_x_index = torch.tensor(neg_x_index, dtype=torch.long)
    neg_y_index = torch.tensor(neg_y_index, dtype=torch.long)
    CDAvalid = torch.from_numpy(CDAvalid).long()
    CDAtest = torch.from_numpy(CDAtest).long()
    lr = args.lr
    weight_decay = args.weight_decay
    alpha = args.alpha
    latent_dim = args.latent_dim
    layer_num = args.layer_num
    epochs = args.epochs

    loss_fun = CDA_PU_loss()

    model = layer_GCN(circRNA_num, disease_num, latent_dim, layer_num)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        circRNA_all_embeddings, disease_all_embeddings, predict_score = model(CD_asso_train)

        loss = loss_fun(predict_score, torch.from_numpy(CD_asso_train), pos_x_index, pos_y_index, neg_x_index,
                        neg_y_index, alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_valid_auc, best_valid_aupr, test_auc, test_aupr, fpr, tpr, precision, recall = evaluate(model,
                                                                                                     CD_asso_train,
                                                                                                     CDAvalid, CDAtest)
        print('Epoch {:d} | Train Loss {:.4f} | best_valid_auc {:.4f} | best_valid_aupr {:.4f} |'
              'test_auc {:.4f} |test_aupr {:.4f}'.format(
            epoch + 1, loss.item(), best_valid_auc, best_valid_aupr, test_auc, test_aupr))
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr, fpr, tpr, precision, recall


class CDA_PU_loss(nn.Module):
    def __init__(self):
        super(CDA_PU_loss, self).__init__()

    def forward(self, CDA_ass_reconstruct, CD_asso_train, pos_x_index, pos_y_index, neg_x_index,
                neg_y_index, alpha):
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss_mat = loss_fn(CDA_ass_reconstruct.float(), CD_asso_train.float())

        loss = (loss_mat[pos_x_index, pos_y_index].sum() * ((1 - alpha) / 2) + loss_mat[
            neg_x_index, neg_y_index].sum() * (alpha / 2))

        return loss


class layer_GCN(nn.Module):
    def __init__(self, circRNA_num, disease_num, latent_dim, n_layers):
        super(layer_GCN, self).__init__()
        self.circRNA_num = circRNA_num
        self.disease_num = disease_num
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.circRNA_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.circRNA_num, self.latent_dim))).type(torch.FloatTensor)
        self.disease_embeddings = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.disease_num, self.latent_dim))).type(torch.FloatTensor)
        self.ego_embeddings = torch.cat([self.circRNA_embeddings, self.disease_embeddings], 0)

        self.tmp = torch.randn(self.latent_dim).type(torch.FloatTensor)
        self.re_CD = nn.Parameter(torch.diag(torch.nn.init.normal_(self.tmp))).type(torch.FloatTensor)  # [64,64]

    def forward(self, A):
        ego_embeddings = self.ego_embeddings
        layer_emdeddings = ego_embeddings
        all_layer_emdeding = list()
        a = torch.from_numpy(relations_to_matrix(A))
        A = re_normalization(a).type(torch.FloatTensor)

        for layer_index in range(self.n_layers):
            layer_emdeddings = torch.sparse.mm(A, layer_emdeddings)
            _weights = F.cosine_similarity(layer_emdeddings, ego_embeddings, dim=-1).unsqueeze(1)
            layer_emdeddings = _weights * layer_emdeddings
            all_layer_emdeding.append(layer_emdeddings)

        ui_all_embeddings = torch.sum(torch.stack(all_layer_emdeding, dim=0), dim=0)
        # print('ui_all_embeddings.size:', ui_all_embeddings.size())
        circRNA_all_embeddings, disease_all_embeddings = torch.split(ui_all_embeddings,
                                                                     [self.circRNA_num, self.disease_num])
        predict_score = torch.mm(torch.mm(circRNA_all_embeddings, self.re_CD), disease_all_embeddings.t())
        return circRNA_all_embeddings, disease_all_embeddings, predict_score


# circRNA_disease_asso, circRNA_sequence_sim, dis_sematic_sim = load_data('./dataset/')
# model = layer_GCN(1491, 208, 64, 3)
# circRNA_all_embeddings, disease_all_embeddings, predict_score = model(circRNA_disease_asso)

def main(args):
    disease_num = 93
    circRNA_num = 625

    test_auc_fold = []
    test_aupr_fold = []
    rs = np.random.randint(0, 1000, 1)[0]
    circRNA_disease_asso, data_set, circRNA_sequence_sim, dis_sematic_sim = load_data('./dataset/', args)

    kf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=rs)
    fold = 1

    for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
        # print('1')
        CDAtrain, CDAtest = data_set[train_index], data_set[test_index]
        CDAtrain, CDAvalid = train_test_split(CDAtrain, test_size=0.05, random_state=rs)
        print("#############%d fold" % fold + "#############")
        fold = fold + 1
        CD_asso_train, pos_x_index, pos_y_index, neg_x_index, neg_y_index = get_train(CDAtrain, circRNA_num,
                                                                                      disease_num)
        best_valid_auc, best_valid_aupr, test_auc, test_aupr, fpr, tpr, precision, recall = train_and_evaluate(
            CD_asso_train, CDAvalid, CDAtest, pos_x_index, pos_y_index, neg_x_index,
            neg_y_index, circRNA_num, disease_num, args)
        test_auc_fold.append(test_auc)
        test_aupr_fold.append(test_aupr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--kfolds', default=10, type=int)
    parser.add_argument('--ratio', default='all', type=str)

    parser.add_argument('--layer_num', default=2, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)

    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--alpha', default=0.6, type=float)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    args = parser.parse_args()
    print(args)
    main(args)
