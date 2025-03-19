import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            init_weights: bool = True,
    ) -> None:
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.layer8 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        self.layer9 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        self.layer10 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True))
        self.layer11 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True))
        self.layer12 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True))
        self.layer13 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True))
        # original: (512, 4096)-(4096, 4096)
        self.layer14 = nn.Sequential(nn.Linear(512, 4096),
                                     nn.BatchNorm1d(4096),
                                     nn.ReLU(inplace=True))
        self.layer15 = nn.Sequential(nn.Linear(4096, 4096),
                                     nn.BatchNorm1d(4096),
                                     nn.ReLU(inplace=True))
        self.layer16 = nn.Linear(4096, num_classes)
        self.flat = nn.Flatten(1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor, get_feature=False, before_act=False):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        x = self.layer5(x)
        f4 = x
        x = self.layer6(x)
        f5 = x
        x = self.layer7(x)
        f6 = x
        x = self.layer8(x)
        f7 = x
        x = self.layer9(x)
        f8 = x
        x = self.layer10(x)
        f9 = x
        x = self.layer11(x)
        f10 = x
        x = self.layer12(x)
        f11 = x
        x = self.layer13(x)
        f12 = x
        x = self.flat(x)
        x = self.layer14(x)
        f13 = x
        x = self.layer15(x)
        x = self.layer16(x)

        if get_feature:
            return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13], x
        else:
            return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG16Pool(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=10):
        super(VGG16Pool, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, get_feature=False, before_act=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)

        if get_feature:
            if before_act:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
