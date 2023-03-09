import torch
import torch.nn as nn
import torch.nn.functional as F


class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()

    def forward(self, x):
        return x.clamp(min=0., max=255.)


class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.act = None

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y


class d_RMBN(nn.Module):
    def __init__(self, colors=3, module_nums=10, channel_nums=32, act_type="prelu", scale=4):
        super(d_RMBN, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.conv0 = nn.Conv2d(self.colors, self.channel_nums, 3, 1, 1)

        backbone = []
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type),
                         ]
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
    model = d_RMBN().cuda()
    y0 = model(x)
    print(y0.shape)
