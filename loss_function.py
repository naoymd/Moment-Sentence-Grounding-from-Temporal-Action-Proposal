import torch
import torch.nn as nn
import math
from utils import l2norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def matmul_loss_function(batch_size, matmul_sim):
    loss_filter = torch.ones(batch_size, dtype=torch.float32) - \
        torch.eye(batch_size, dtype=torch.float32)
    loss_filter = loss_filter.to(device)
    matmul_loss = torch.mul(matmul_sim, loss_filter).to(device)
    matmul_loss = torch.abs(matmul_loss).to(device)
    matmul_loss = l2norm(matmul_loss)
    matmul_loss = torch.mean(matmul_loss).to(device)
    return matmul_loss


def cos_loss_function(cos_sim):
    ones = torch.ones_like(cos_sim, dtype=torch.float32).to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    cos_loss = mse_loss(ones, cos_sim)
    # l1_loss = torch.nn.L1Loss().to(device)
    # cos_loss = l1_loss(ones, cos_sim)
    return cos_loss


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1e-2, method='sum', improved=False, intra=0.5, lamb=1e-2):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.method = method
        self.improved = improved
        self.intra = intra
        self.lamb = lamb

    # im, sen : (n_samples, dim)
    def forward(self, matmul_sim):
        # print('rank loss', matmul_sim.size())

        # matmul_sim = matmul_sim / math.sqrt(common_size)
        matmul_size = matmul_sim.size(0)

        positive = matmul_sim.diag().view(-1, 1)
        positive_sen = positive.expand_as(matmul_sim)
        positive_vid = positive.t().expand_as(matmul_sim)
        # print(positive_sen.size(), positive_vid.size())

        # mask for diagonals
        mask = (torch.eye(matmul_size) > 0.5).to(matmul_sim.device)
        # loss_mat : (n_samples, n_samples)
        # video negatives
        loss_mat_sen = (self.margin + matmul_sim -
                        positive_sen).clamp(min=0).masked_fill(mask, 0)
        # sentence negatives
        loss_mat_vid = (self.margin + matmul_sim -
                        positive_vid).clamp(min=0).masked_fill(mask, 0)
        # sum of hinges loss
        if self.method == "sum":
            pass
        # max of hinges loss
        elif self.method == "max":
            # lossmat : (n_samples)
            loss_mat_sen = loss_mat_sen.max(dim=1)[0]
            loss_mat_vid = loss_mat_vid.max(dim=0)[0]

        loss = loss_mat_sen.sum() + loss_mat_vid.sum()

        if self.improved:
            loss += self.lamb * \
                ((self.intra - matmul_sim.diag()).clamp(min=0).sum())

        if self.method == "sum":
            matmul_dim = torch.tensor(matmul_sim.size(
                0), dtype=torch.float32).to(device)
            # return loss / torch.sqrt(matmul_dim)
            return loss / matmul_dim

        elif self.method == "max":
            return loss
