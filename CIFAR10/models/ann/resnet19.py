import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OneWay(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None):
        super(OneWay, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        x = self.conv(x)
        x = self.bn(x) 
        x = self.relu(x)
        return x


class TwoWays(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, downsample=None):
        super(TwoWays, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.downsample = downsample
        self.idem = False

    def forward(self, x, identity):
        if self.idem:
            return x
        x = self.conv(x)
        x = self.bn(x) 
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x   


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.idem = False
        self.inter = False

        self.part1 = OneWay(conv3x3(inplanes, planes, stride),
                            norm_layer(planes),
                            nn.ReLU(inplace=True))
        
        self.part2 = TwoWays(conv3x3(planes, planes),
                             norm_layer(planes),
                             nn.ReLU(inplace=True), downsample)

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        out = self.part2(out, identity)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.layer0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = nn.Sequential(nn.AvgPool2d(8, stride=1),
                                    nn.Flatten(1),
                                    nn.Linear(512, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, get_feature=False):
        hidden_layers = []

        x = self.layer0(x)
        hidden_layers.append(x)
        x = self.layer1(x)
        hidden_layers.append(x)
        x = self.layer2(x)
        hidden_layers.append(x)
        x = self.layer3(x)
        hidden_layers.append(x)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == 1:
                hidden_layers.append(x)

        if get_feature:
            return hidden_layers, x
        else:
            return x


def resnet19(**kwargs):
    return ResNet(BasicBlock, [3, 3, 2], **kwargs)
