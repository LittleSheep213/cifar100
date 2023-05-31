from torch.nn import functional as F
from torch import nn
import torch


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1)).reshape(-1, 100)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def topN_count(pred, gt, N):
    """
    :param pred: 网络的输出
    :param gt: 真实标签
    :param N: 猜测个数
    :return: 猜中个数
    """
    import torch.nn as nn
    sm = nn.Softmax(dim=0)
    prob = sm(pred)
    top_list = []
    for n in range(N):
        top_n = prob.argmax(1)
        for row, i in enumerate(top_n):
            prob[row, i] = 0
        top_list.append(top_n.reshape(-1, 1))
    res = torch.cat(top_list, dim=1)

    #
    gt = gt.reshape(-1, 1)

    # 计数
    count = 0
    for i, l in enumerate(gt):
        if l in res[i]:
            count += 1
    return count
