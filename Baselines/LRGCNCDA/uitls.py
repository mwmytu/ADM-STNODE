# -*- coding: utf-8 -*-
# @Time : 2022/9/3 | 18:43
# @Author : YangCheng
# @Email : yangchengyjs@163.com
# @File : uitls.py
# Software: PyCharm

import numpy as np
import torch

def load_data(data_path, args):
    #circRNA_disease_asso = np.loadtxt(data_path + 'circRNA_disease_association.csv', delimiter=',')  # [1491,208]
    circRNA_disease_asso=np.loadtxt(data_path+'associationMatrix_625_93.csv',delimiter=',')
    circRNA_sequence_sim = np.loadtxt(data_path + 'circ_sequence_similarity.csv', delimiter=',')  # [1491,1491]
    dis_sematic_sim = np.loadtxt(data_path + 'dis_semantic_similarity_CircR2D_208.csv', delimiter=',')  # [208,208]

    '''
        preparing for 5 cross-validation
        '''
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(circRNA_disease_asso)[0]):
        for j in range(np.shape(circRNA_disease_asso)[1]):
            if int(circRNA_disease_asso[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(circRNA_disease_asso[i][j]) == 0:
                whole_negative_index.append([i, j])

    if args.ratio == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=10 * len(whole_positive_index), replace=False)
    elif args.ratio == 'one':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)
    elif args.ratio == 'all':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)), size=len(whole_negative_index),
                                                 replace=False)
    else:
        print('wrong positive negative ratio')

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    # print (data_set)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    return circRNA_disease_asso, data_set, circRNA_sequence_sim, dis_sematic_sim


def re_normalization(A):
    A = A + torch.eye(A.size()[0])
    deg = torch.sum(A, dim=1)
    deg_inv = deg.pow(-0.5) * torch.eye(A.size()[0])
    A = torch.mm(torch.mm(deg_inv, A), deg_inv)
    return A


def relations_to_matrix(circRNA_disease_asso):
    m, n = circRNA_disease_asso.shape[0], circRNA_disease_asso.shape[1]
    circRNA_circRNA_part = np.zeros((m, m))
    diease_disease_part = np.zeros((n, n))
    h1 = np.hstack((circRNA_circRNA_part, circRNA_disease_asso))
    h2 = np.hstack((circRNA_disease_asso.T, diease_disease_part))
    A = np.vstack((h1, h2))
    return A
