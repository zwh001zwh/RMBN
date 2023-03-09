import numpy as np
import torch
import torch.nn as nn
from archs.RMBM import *
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from archs.RMBM1 import *


class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()

    def forward(self, x):
        return x.clamp(min=0., max=255.)


@ARCH_REGISTRY.register()
class t_RMBN(nn.Module):
    def __init__(self, colors=3, module_nums=10, channel_nums=32, with_idt=False, act_type="prelu", scale=4):
        super(t_RMBN, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.conv0 = nn.Conv2d(self.colors, self.channel_nums, 3, 1, 1)

        backbone = []
        for i in range(self.module_nums):
            backbone += [RMBM(self.channel_nums, self.channel_nums, act_type=self.act_type,
                              with_idt=self.with_idt)]
        self.conv1 = nn.Conv2d(self.channel_nums, self.colors * self.scale * self.scale, 3, 1, 1)
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
        self.clip = ClippedReLU()

    def forward(self, x):
        shortcut = torch.repeat_interleave(x, self.scale * self.scale, dim=1)
        x = self.conv0(x)
        y = self.backbone(x) + x
        y = self.conv1(y)
        y = y + shortcut
        y = self.upsampler(y)
        y = self.clip(y)
        return y


if __name__ == '__main__':
    x = torch.randn(1, 3, 5, 5).cuda()
    model = t_RMBN().cuda()
    y0 = model(x)
    print(y0.shape)
