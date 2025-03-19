import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_st, y_ta):
        p_st = F.log_softmax(y_st/self.T, dim=1)
        p_ta = F.softmax(y_ta/self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_st, p_ta) * (self.T**2)
        return loss
