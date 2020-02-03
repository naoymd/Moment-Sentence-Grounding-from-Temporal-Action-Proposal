# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import math
from utils import l2norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ground(nn.Module):
    def __init__(self, common_size, mode='simple'):
        super(Ground, self).__init__()
        self.mode = mode
        self.multi_linear = nn.Linear(common_size*2, common_size)
        self.multi_dropout = nn.Dropout(p=0.2)
        self.sen_linear = nn.Linear(common_size*2, common_size)
        self.sen_dropout = nn.Dropout(p=0.1)
        self.vid_linear = nn.Linear(common_size*2, common_size)
        self.vid_dropout = nn.Dropout(p=0.1)
        init.xavier_uniform_(self.multi_linear.weight)
        init.xavier_uniform_(self.sen_linear.weight)
        init.xavier_uniform_(self.vid_linear.weight)

    def forward(self, sen2vec, vid2vec):
        # print('sen2vec.size: {}, vid2vec.size: {}'.format(
        #     sen2vec.size(), vid2vec.size()))
        if self.mode == 'simple':
            matmul_sim = self._matmul_similarity(sen2vec, vid2vec)
            cos_sim = self._cos_similarity(sen2vec, vid2vec)

        elif self.mode == 'multi':
            if sen2vec.size() != vid2vec.size():
                sen2vec = sen2vec.repeat_interleave(vid2vec.size(0), dim=0)
            multi_vec = torch.cat((sen2vec, vid2vec), dim=1)
            multi_vec = self.multi_linear(multi_vec)
            multi_vec = self.multi_dropout(multi_vec)
            multi_vec = l2norm(multi_vec)
            multi_sen2vec = torch.cat((sen2vec, multi_vec), dim=1)
            multi_sen2vec = self.sen_linear(multi_sen2vec)
            multi_sen2vec = self.sen_dropout(multi_sen2vec)
            multi_sen2vec = l2norm(multi_sen2vec)
            multi_vid2vec = torch.cat((vid2vec, multi_vec), dim=1)
            multi_vid2vec = self.vid_linear(multi_vid2vec)
            multi_vid2vec = self.vid_dropout(multi_vid2vec)
            multi_vid2vec = l2norm(multi_vid2vec)

            matmul_sim = torch.stack((self._matmul_similarity(
                sen2vec, vid2vec), self._matmul_similarity(multi_sen2vec, multi_vid2vec)), dim=0)
            matmul_sim = torch.mean(matmul_sim, dim=0)
            cos_sim = torch.stack((self._cos_similarity(
                sen2vec, vid2vec), self._cos_similarity(multi_sen2vec, multi_vid2vec)), dim=0)
            cos_sim = torch.mean(cos_sim, dim=0)

        else:
            matmul_sim = self._matmul_similarity(sen2vec, vid2vec)
            cos_sim = self._cos_similarity(sen2vec, vid2vec)
        # print('matmal, cos', matmul_sim.size(), cos_sim.size())

        return matmul_sim, cos_sim

    def _matmul_similarity(self, sen2vec, vid2vec):
        matmul_sim = torch.matmul(sen2vec, vid2vec.t())
        matmul_size = torch.tensor([matmul_sim.size(0)],
                                   dtype=torch.float32).to(device)
        # matmul_sim = matmul_sim / torch.sqrt(matmul_size)
        return matmul_sim

    def _cos_similarity(self, sen2vec, vid2vec):
        cos_similarity = torch.nn.CosineSimilarity(dim=1).to(device)
        cos_sim = cos_similarity(sen2vec, vid2vec)
        # print(cos_sim)
        return cos_sim
