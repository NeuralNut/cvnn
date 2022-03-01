# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:39:57 2022

@author: mlsol
"""

import torch as t
import torchlayers as tl
from cvnn.layers import Linear, ComplexLinear
from cvnn.layers import ComplexConv2D, ComplexBatchNorm2D
from cvnn.layers import ComplexReLU
from cvnn.layers import ComplexMaxPool2D, ComplexAdapAvgPool2D, GlobalAveragePooling2D
from cvnn.layers import ComplexInverseDropout, ComplexDropout, ComplexDropout2D
from cvnn.layers import Abs, Intensity, Magnitude


class CVBlock(t.nn.Module):
    def __init__(self, num_layers, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(CVBlock, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = ComplexConv2D(out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = ComplexBatchNorm2D()
        if self.num_layers > 34:
            self.conv2 = ComplexConv2D(out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = ComplexConv2D(out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = ComplexBatchNorm2D()
        self.conv3 = ComplexConv2D(out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = ComplexBatchNorm2D()
        self.relu = ComplexReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class CVResNet(t.nn.Module):
  # https://niko-gamulin.medium.com/resnet-implementation-with-pytorch-from-scratch-23cf3047cb93
    def __init__(self, num_layers, out_dim, c_to_r, mode=None, block=CVBlock, 
                 i_chans=[64, 64, 128, 256, 512], img_shape=None, target_shape=t.tensor([224,224]) ):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(CVResNet, self).__init__()
        self.i_chans = i_chans
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        
        #TODO: input padding to at least 197
        shape_diff = target_shape - img_shape
        shape_diff = [max(1,i) for i in shape_diff]
        L, R, T, B = shape_diff[0]//2+1, shape_diff[0]//2+1, shape_diff[1]//2+1, shape_diff[1]//2+1

        self.input_padding = t.nn.ZeroPad2d([L, R, T, B])
        self.conv1 = ComplexConv2D(self.i_chans[0], kernel_size=7, stride=2, padding=3) # 64
        self.bn1 = ComplexBatchNorm2D()
        self.relu = ComplexReLU()
        self.maxpool = ComplexMaxPool2D(kernel_size=3,stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=self.i_chans[1], stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=self.i_chans[2], stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=self.i_chans[3], stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=self.i_chans[4], stride=2)

        self.avgpool = ComplexAdapAvgPool2D((1, 1))
        self.flatten = t.nn.Flatten()
        self.c_to_r = c_to_r
        self.fc = Linear(out_dim)
        self.mode = mode
        self.softmax = t.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.input_padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.c_to_r(x)
        x = self.fc(x)
        if self.mode=='classifier':
            x = self.softmax(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = t.nn.ModuleList()

        identity_downsample = t.nn.Sequential(ComplexConv2D(intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                              ComplexBatchNorm2D())
        layers.append(block(num_layers, intermediate_channels, identity_downsample, stride))
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return t.nn.Sequential(*layers)
    
def CVResNet18(num_classes=1000):
    return CVResNet(18, CVBlock, num_classes)


def CVResNet34(num_classes=1000):
    return CVResNet(34, CVBlock, num_classes)


def CVResNet50(num_classes=1000):
    return CVResNet(50, CVBlock, num_classes)


def CVResNet101(num_classes=1000):
    return CVResNet(101, CVBlock, num_classes)


def CVResNet152(num_classes=1000):
    return CVResNet(152, CVBlock, num_classes)


class RVBlock(t.nn.Module):
    def __init__(self, num_layers, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(RVBlock, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = tl.Conv2d(out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = tl.BatchNorm2d()
        if self.num_layers > 34:
            self.conv2 = tl.Conv2d(out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = tl.Conv2d(out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = tl.BatchNorm2d()
        self.conv3 = tl.Conv2d(out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = tl.BatchNorm2d()
        self.relu = tl.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class RVResNet(t.nn.Module):
  # https://niko-gamulin.medium.com/resnet-implementation-with-pytorch-from-scratch-23cf3047cb93
    def __init__(self, num_layers, out_dim, mode=None, block=RVBlock, 
                 i_chans=[64, 64, 128, 256, 512], img_shape=None, target_shape=(224,224)):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(RVResNet, self).__init__()
        self.i_chans = i_chans
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        
        #TODO: input padding to at least 197
        # shape_diff = img_shape - target_shape
        # self.input_padding = t.nn.ZeroPad2d()
        self.conv1 = tl.Conv2d(self.i_chans[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = tl.BatchNorm2d()
        self.relu = tl.ReLU()
        self.maxpool = tl.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=self.i_chans[1], stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=self.i_chans[2], stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=self.i_chans[3], stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=self.i_chans[4], stride=2)

        self.avgpool = tl.AdaptiveAvgPool2d((1, 1))
        self.flatten = t.nn.Flatten()
        self.fc = Linear(out_dim)
        self.mode = mode
        self.softmax = t.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x = self.input_padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        if self.mode=='classifier':
            x = self.softmax(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = t.nn.ModuleList()

        identity_downsample = t.nn.Sequential(tl.Conv2d(intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                              tl.BatchNorm2d())
        layers.append(block(num_layers, intermediate_channels, identity_downsample, stride))
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return t.nn.Sequential(*layers)
    
    
def CVFCN(ofeats_per_block, ksize_per_block, *args, **kwargs):
    """

    :param ofeats_per_block: a list of integers, the number of output features for each convolutional
    block
    :param ksize_per_block: A list of kernel sizes for each block

    """

    def complex_conv_block(out_features, *args, **kwargs):
        return t.nn.Sequential(
            ComplexConv2D(out_features, *args, **kwargs),
            ComplexBatchNorm2D(),
            ComplexReLU(),
        )

    net = []
    for i, ofeats in enumerate(ofeats_per_block):
        net.append(complex_conv_block(ofeats, kernel_size=ksize_per_block[i]))
    net.append(GlobalAveragePooling2D())
    return t.nn.Sequential(*net)


def CVMLP(ofeats_per_block, dropout_per_block, *args, **kwargs):
    """
    Cannot generate summary

    Args:
      ofeats_per_block: a list of integers, the number of output features for each block.
      dropout_per_block: a list of dropout rates for each block.

    Returns:
      A list of ComplexLinear, ComplexReLU, ComplexInverseDropout, and Flatten.
    """
    net = []
    net.append(t.nn.Flatten())
    for i, ofeats in enumerate(ofeats_per_block):
        net.append(ComplexLinear(ofeats))
        net.append(ComplexReLU())
        net.append(ComplexInverseDropout(dropout_per_block[i]))
    return t.nn.Sequential(*net)

