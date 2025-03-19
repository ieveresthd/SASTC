import torch
import torch.nn as nn


class SASTCLoss(nn.Module):
    """Self attention based temporal knowledge distillation"""

    def __init__(self):
        super(SASTCLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, s_value, f_target, weight):
        weight = torch.stack(weight)
        timesteps, bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(timesteps, bsz, num_stu, num_tea).cuda()

        for t in range(timesteps):
            for i in range(num_stu):
                for j in range(num_tea):
                    ind_loss[t, :, i, j] = self.crit(s_value[t][i][j], f_target[t][i][j]).reshape(bsz, -1).mean(-1)
        loss_time = (weight * ind_loss).sum() / (1.0 * bsz * timesteps * num_stu)
        return loss_time
