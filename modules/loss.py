import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weight=None, epsilon: float = 0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def linear_combination(self, x, y, epsilon):
        return epsilon*x + (1-epsilon)*y

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, self.weight, reduction=self.reduction)
        return self.linear_combination(loss/n, nll, self.epsilon)


class InstanceLossBoost(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, temperature, cluster_num, device):
        super().__init__()
        self.temperature = temperature
        self.cluster_num = cluster_num
        self.device = device

 
    def forward(self, z_i, z_j, pseudo_label):
        n = z_i.shape[0]
        invalid_index = pseudo_label == -1  # 没有为标签的数据的索引
        
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to( self.device )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(self.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()
        
        mask = mask.repeat(2, 2)
        mask_eye = mask_eye.repeat(2, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n*2).view(-1, 1).to(self.device),
            0,
        )
        logits_mask *= 1 - mask  # 负例对
        mask_eye = mask_eye * logits_mask  # 正例对

        z = torch.cat((z_i, z_j), dim=0) 
        sim = torch.matmul(z, z.t()) / self.temperature  # z @ z.t() / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)  # 获取每一行的最大值, 并保持2*n行1列
        sim = sim - sim_max.detach()  #  这样做是为了防止上溢，因为后面要进行指数运算

        exp_sim_neg = torch.exp(sim) * logits_mask  # 得到只有负例相似对的矩阵
        log_sim = sim - torch.log(exp_sim_neg.sum(1, keepdim=True))  #  log_softmax(), 分子上 正负例对 都有

        # compute mean of log-likelihood over positive
        instance_loss = -(mask_eye * log_sim).sum(1) / mask_eye.sum(1)  # 去分子为正例对的数据 
        instance_loss = instance_loss.view(2, n).mean()

        return instance_loss


class ClusterLossBoost(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, cluster_num, device):
        super().__init__()
        self.cluster_num = cluster_num
        self.device = device

    def forward(self, c_j, pseudo_label_all, index):
        pseudo_label = pseudo_label_all[index]  # 当前miniBatch的数据伪标签

        # 获取每个类别的权重
        pseudo_index_all = pseudo_label_all != -1
        pseudo_label_all = pseudo_label_all[pseudo_index_all]
        idx, counts = torch.unique(pseudo_label_all, return_counts=True)
        freq = pseudo_label_all.shape[0] / counts.float()
        weight = torch.ones(self.cluster_num).to(self.device)
        weight[idx] = freq

        # 构建自标签（self-label learning) 损失函数
        pseudo_index = pseudo_label != -1  # 这里需要更改！！！！！
        
        if pseudo_index.sum() > 0:
            criterion = LabelSmoothingCrossEntropy(weight=weight).to(self.device)  # 默认reduction是求平均
            # criterion = nn.CrossEntropyLoss(weight=weight).to(self.device)  # 默认reduction是求平均
            loss_ce = criterion(
                c_j[pseudo_index], pseudo_label[pseudo_index].to(self.device)
            )
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).to(self.device)
        return loss_ce



