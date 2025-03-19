import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import *


class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class VGG11SNN(nn.Module):
    def __init__(self, T):
        super(VGG11SNN, self).__init__()
        self.T = T
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.layer1 = Layer(3, 64, 3, 1, 1)
        self.layer2 = Layer(64, 128, 3, 2, 1)
        # pool,
        self.layer3 = Layer(128, 256, 3, 1, 1)
        self.layer4 = Layer(256, 256, 3, 2, 1)
        # pool,
        self.layer5 = Layer(256, 512, 3, 1, 1)
        self.layer6 = Layer(512, 512, 3, 2, 1)
        # pool,
        self.layer7 = Layer(512, 512, 3, 1, 1)
        self.layer8 = Layer(512, 512, 3, 2, 1)
        # pool,
        self.layer9 = SeqToANNContainer(nn.Linear(512 * 2 * 2, 4096))
        self.layer10 = SeqToANNContainer(nn.Linear(4096, 4096))
        self.layer11 = SeqToANNContainer(nn.Linear(4096, 10))

    def forward(self, input, get_feature=False):
        hidden_layers = []

        x = add_dimention_distribute(input, self.T)

        x = self.layer1(x)
        hidden_layers.append(x)
        x = self.layer2(x)
        hidden_layers.append(x)
        x = self.layer3(x)
        x = self.layer4(x)
        hidden_layers.append(x)
        x = self.layer5(x)
        x = self.layer6(x)
        hidden_layers.append(x)
        x = self.layer7(x)
        x = self.layer8(x)
        hidden_layers.append(x)

        x = torch.flatten(x, 2)
        x = self.layer9(x)
        x = self.layer10(x)
        hidden_layers.append(x)
        x = self.layer11(x)

        if get_feature:
            return hidden_layers, x
        else:
            return x
