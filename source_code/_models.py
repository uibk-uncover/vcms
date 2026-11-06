"""Networks: the cost generator (Alice) and the UCNet-based detector (Eve).

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import torch
from torch import nn
import torch.nn.functional as F

from _srm_kernel_filters import build_filters


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 27,
        mid_channels: int = 192,
        kernel_size: int = 7,
    ):
        super(DepthwiseSeparableBlock, self).__init__()
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                groups=in_channels,
                bias=False,
                padding_mode='reflect',
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=1,
                bias=True
            )
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.conv1(x0)


class Block2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),  # kernel size not in the paper
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=1),
        )
    def forward(self, x):
        x = F.leaky_relu(self.basic(x) + self.shortcut(x), negative_slope=.2)
        return x


class Block3(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=.2),
            nn.Conv2d(out_channels, out_channels, stride=2, groups=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.basic(x) + self.shortcut(x), negative_slope=.2)
        return x


class UcNetD(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(UcNetD, self).__init__()
        # === preprocessing module ===
        self.preprocessing = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 62, groups=in_channels, kernel_size=5, padding=2, bias=False),
            nn.Hardtanh(min_val=-2, max_val=2))  # TLU
        filters = torch.Tensor(build_filters()).view(62, 1, 5, 5).repeat(in_channels, 1, 1, 1)
        with torch.no_grad():
            self.preprocessing[0].weight.copy_(filters)  # hard-set filters
        self.preprocessing[0].weight.requires_grad = False  # non-trainable
        # === convolutional module ===
        self.block_1a = nn.Sequential(
            nn.Conv2d(in_channels*62, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2),
        )
        self.block_2a = Block2(32, 32)
        self.block_3 = Block3(32, 64)
        self.block_2b = Block2(64, 128)
        self.block_1b = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=.2),
        )
        # === classification module ===
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(256, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # preprocessing module
        x = self.preprocessing(x * 255.)
        # convolutional module
        x = self.block_1a(x)
        x = self.block_2a(x)
        x = self.block_3(x)
        x = self.block_2b(x)
        x = self.block_1b(x)
        # classification module
        x = self.global_pool(x)
        y = self.classifier(x)
        return y
