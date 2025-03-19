import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionTime(nn.Module):
    """Cross layer Self Attention per time step"""

    def __init__(self, s_len, t_len, input_channel, s_n, s_t, factor=4):
        super(SelfAttentionTime, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for i in range(t_len):
            setattr(self, 'key_weight' + str(i), MLPEmbed(input_channel, input_channel // factor))
        for i in range(s_len):
            setattr(self, 'query_weight' + str(i), MLPEmbed(input_channel, input_channel // factor))

        for i in range(s_len):
            for j in range(t_len):
                setattr(self, 'regressor' + str(i) + str(j), AAEmbed(s_n[i], s_t[j]))

    def forward(self, feat_s, feat_t):
        timesteps = feat_s[0].size(1)
        sim_t = list(range(len(feat_t)))
        # sim_s = list(range(len(feat_s)))
        bsz = feat_s[0].shape[0]
        sim_s = torch.empty((timesteps, len(feat_s), bsz, bsz))
        # similarity matrix per time step
        for i in range(len(feat_t)):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())
        for t in range(timesteps):
            for i in range(len(feat_s)):
                sim_temp = feat_s[i][:, t, ...].reshape(bsz, -1)
                sim_s[t][i] = torch.matmul(sim_temp, sim_temp.t())
        sim_s = sim_s.cuda()
        # key of target layers
        proj_key = self.key_weight0(sim_t[0])
        proj_key = proj_key[:, :, None]

        for i in range(1, len(sim_t)):
            temp_proj_key = getattr(self, 'key_weight' + str(i))(sim_t[i])
            proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)

        # query of source layers
        proj_query = list(range(timesteps))
        for t in range(timesteps):
            proj_query[t] = self.query_weight0(sim_s[t][0])
            proj_query[t] = proj_query[t][:, None, :]
            for i in range(1, sim_s.size(1)):
                temp_proj_query = getattr(self, 'query_weight' + str(i))(sim_s[t][i])
                proj_query[t] = torch.cat([proj_query[t], temp_proj_query[:, None, :]], 1)

        # attention weight
        energy = list(range(timesteps))
        attention = list(range(timesteps))
        for t in range(timesteps):
            energy[t] = torch.bmm(proj_query[t], proj_key)  # batch_size X No.stu feature X No.tea feature
            attention[t] = F.softmax(energy[t], dim=-1)

        # feature space alignment
        for i in range(len(feat_s)):
            feat_s[i] = feat_s[i].permute(1, 0, 2, 3, 4)
        proj_value_stu_time = []
        value_tea_time = []
        for t in range(timesteps):
            proj_value_stu_time.append([])
            value_tea_time.append([])
            for i in range(len(feat_s)):
                proj_value_stu_time[t].append([])
                value_tea_time[t].append([])
                for j in range(len(feat_t)):
                    s_H, t_H = feat_s[i].shape[3], feat_t[j].shape[2]
                    if s_H > t_H:
                        # input = F.adaptive_avg_pool2d(feat_s[i][t], (t_H, t_H))
                        input = F.adaptive_max_pool2d(feat_s[i][t], (t_H, t_H))
                        proj_value_stu_time[t][i].append(getattr(self, 'regressor' + str(i) + str(j))(input))
                        value_tea_time[t][i].append(feat_t[j])
                    elif s_H < t_H or s_H == t_H:
                        # target = F.adaptive_max_pool2d(feat_t[j], (s_H, s_H))
                        target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
                        proj_value_stu_time[t][i].append(getattr(self, 'regressor' + str(i) + str(j))(feat_s[i][t]))
                        value_tea_time[t][i].append(target)

        return proj_value_stu_time, value_tea_time, attention


class AAEmbed(nn.Module):
    """non-linear embed by MLP"""

    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        def conv5x5(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=1, stride=stride, bias=False)

        def conv7x7(in_channels, out_channels, stride=3):
            return nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv3x3(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
            # nn.BatchNorm2d(self.num_mid_channel),
            # nn.ReLU(inplace=True),
            # conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""

    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super(PoolEmbed, self).__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels * pool_size * pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)
