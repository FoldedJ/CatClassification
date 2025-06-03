import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)

        # 调整通道顺序以匹配LayerNorm的期望
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class LayerNorm2d(nn.Module):
    """将LayerNorm应用到通道维度上的包装器"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


class MyConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]

        self.downsample_layers = nn.ModuleList()

        # 第一个下采样层（stem）
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])  # 使用自定义的LayerNorm2d
        )
        self.downsample_layers.append(stem)

        # 其余下采样层
        for i in range(3):
            downsample = nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                LayerNorm2d(dims[i + 1])  # 使用自定义的LayerNorm2d
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-2, -1])  # global average pooling
        x = self.norm(x)
        x = self.head(x)
        return x


def build_model(num_classes):
    return MyConvNeXt(num_classes)