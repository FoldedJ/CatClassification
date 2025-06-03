import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


# 定义通道注意力模块（SENet）
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedResNet18, self).__init__()
        # 加载预训练的resnet18
        self.resnet = models.resnet18(pretrained=True)

        # 保持特征提取部分不变
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1

        # 在layer2和layer3中添加注意力机制
        self.layer2 = self._create_layer_with_se(self.resnet.layer2, 128)
        self.layer3 = self._create_layer_with_se(self.resnet.layer3, 256)
        self.layer4 = self.resnet.layer4

        self.avgpool = self.resnet.avgpool
        self.fc = nn.Linear(512, num_classes)

    def _create_layer_with_se(self, original_layer, in_channels):
        new_layer = nn.Sequential()
        for i, module in enumerate(original_layer):
            if i == 0:  # 只在每个layer的第一个block后添加SEBlock
                new_layer.add_module(f"block{i}", nn.Sequential(
                    module,
                    SEBlock(in_channels)
                ))
            else:
                new_layer.add_module(f"block{i}", module)
        return new_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def build_model(num_classes):
    return ImprovedResNet18(num_classes)