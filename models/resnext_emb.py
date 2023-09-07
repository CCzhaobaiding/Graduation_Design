# 加embedding

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class BatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                 cardinality, base_width, widen_factor):

        super().__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D, momentum=0.001)
        self.conv_conv = nn.Conv2d(D, D,
                                   kernel_size=3, stride=stride, padding=1,
                                   groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D, momentum=0.001)
        self.act = mish
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels, momentum=0.001)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=1,
                                               stride=stride,
                                               padding=0,
                                               bias=False))
            self.shortcut.add_module(
                'shortcut_bn', nn.BatchNorm2d(out_channels, momentum=0.001))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = self.act(self.bn_reduce.forward(bottleneck))
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = self.act(self.bn.forward(bottleneck))
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return self.act(residual + bottleneck)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class CifarResNeXt(nn.Module):
    def __init__(self, cardinality, depth, num_classes,
                 base_width, widen_factor=4):

        super().__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 *
                       self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64, momentum=0.001)
        self.act = mish
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        low_dim = 64
        self.classifier = nn.Linear(self.stages[3], num_classes)
        self.l2norm = Normalize(2)
        self.fc = nn.Linear(self.stages[3], low_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def block(self, name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels,
                                                          out_channels,
                                                          pool_stride,
                                                          self.cardinality,
                                                          self.base_width,
                                                          self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels,
                                                   out_channels,
                                                   1,
                                                   self.cardinality,
                                                   self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = self.act(self.bn_1.forward(x))
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, self.stages[3])
        output = self.classifier(x)
        feat = self.fc(x)
        feat = self.l2norm(feat)
        return output, feat


def build_resnext(cardinality, depth, width, num_classes):
    logger.info(f"Model: ResNeXt {depth+1}x{width}")
    return CifarResNeXt(cardinality=cardinality,
                        depth=depth,
                        base_width=width,
                        num_classes=num_classes)
