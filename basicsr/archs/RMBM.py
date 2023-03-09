import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == '1_3_1_2c':
            self.mid_planes = int(inp_planes * 2)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

            conv2 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.k2 = conv2.weight
            self.b2 = conv2.bias
            self.in1 = nn.InstanceNorm2d(num_features=self.mid_planes, affine=True, track_running_stats=True)

        elif self.type == '1_3_1_3c':
            self.mid_planes = int(inp_planes * 3)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

            conv2 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.k2 = conv2.weight
            self.b2 = conv2.bias
            self.in1 = nn.InstanceNorm2d(num_features=self.mid_planes, affine=True, track_running_stats=True)

        elif self.type == '1_3_1_c/2':
            self.mid_planes = int(inp_planes / 2)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

            conv2 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.k2 = conv2.weight
            self.b2 = conv2.bias
            self.in1 = nn.InstanceNorm2d(num_features=self.mid_planes, affine=True, track_running_stats=True)

        elif self.type == '1_3_2c':
            self.mid_planes = int(out_planes * 2)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            self.in1 = nn.InstanceNorm2d(num_features=self.out_planes, affine=True, track_running_stats=True)

        else:
            self.mid_planes = int(out_planes * 3)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            self.in1 = nn.InstanceNorm2d(num_features=self.out_planes, affine=True, track_running_stats=True)

    def forward(self, x):
        if self.type == '1_3_1_2c' or self.type == '1_3_1_c/2' or self.type == '1_3_1_3c':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0_0 = y0
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            # print(y0.shape)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad

            # conv-3x3
            y0_1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
            y0_1 = y0_1 + y0_0
            # conv-1x1
            y1 = F.conv2d(input=y0_1, weight=self.k2, bias=self.b2, stride=1)

        else:
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)

        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == '1_3_1_2c' or self.type == '1_3_1_c/2' or self.type == '1_3_1_3c':
            K_idt1 = torch.zeros(self.mid_planes, self.mid_planes, 3, 3, device=device)
            for i in range(self.mid_planes):
                K_idt1[i, i, 1, 1] = 1.0

            k1 = self.k1 + K_idt1

            # re-param conv kernel
            RK0 = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB0 = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB0 = F.conv2d(input=RB0, weight=k1).view(-1, ) + self.b1

            weight_3_3_ = RK0.clone()
            weight_1_1_ = self.k2.data.clone()
            bias_3_3_ = RB0.data.clone()
            bias_1_1 = self.b2.data.clone()
            reweight_3_3 = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1,
                                     padding=1, bias=True).to(device)
            RK = reweight_3_3.weight.data.clone()
            RB = reweight_3_3.bias.data.clone()
            for i in range(weight_1_1_.shape[0]):
                RK[i, ...] = torch.sum(weight_3_3_ * weight_1_1_[i, ...].unsqueeze(1), dim=0)
                RB[i] = bias_1_1[i] + torch.sum(bias_3_3_ * weight_1_1_[i, ...].squeeze(1).squeeze(1))

        else:
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1

        return RK, RB


class RMBM(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='linear', with_idt=False):
        super(RMBM, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv1_3_1_3c = SeqConv3x3('1_3_1_3c', self.inp_planes, self.out_planes)
        self.conv1_3_1_2c = SeqConv3x3('1_3_1_2c', self.inp_planes, self.out_planes)
        self.conv1_3_1_c2 = SeqConv3x3('1_3_1_c/2', self.inp_planes, self.out_planes)
        self.conv1_3_2c = SeqConv3x3('1_3_2c', self.inp_planes, self.out_planes)
        self.conv1_3_3c = SeqConv3x3('1_3_3c', self.inp_planes, self.out_planes)

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

        a = self.conv1_3_1_2c(x)
        b = self.conv1_3_1_c2(x)
        c = self.conv1_3_2c(x)
        d = self.conv1_3_3c(x)
        e = self.conv1_3_1_3c(x)
        y = a + b + c + d + e

        if self.with_idt:
            y += x
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        K0, B0 = self.conv1_3_1_2c.rep_params()
        K1, B1 = self.conv1_3_1_c2.rep_params()
        K2, B2 = self.conv1_3_2c.rep_params()
        K3, B3 = self.conv1_3_3c.rep_params()
        K4, B4 = self.conv1_3_1_3c.rep_params()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


def trans_fusein(conv_weight, norm):
    weight = norm.weight
    bias = norm.bias
    std = (norm.running_var + norm.eps).sqrt()
    t = (weight / std).reshape(-1, 1, 1, 1)
    return conv_weight * t, bias - norm.running_mean * weight / std


if __name__ == '__main__':
    # # test seq-conv
    # x = torch.randn(1, 3, 5, 5).cuda()
    # conv = SeqConv3x3('1_3_3c', 3, 3).cuda()
    # conv.eval()
    # # print(conv.type)
    # y0 = conv(x)
    # print(y0.shape)
    # RK, RB = conv.rep_params()
    # y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    # print(y1.shape)
    # print(y0 - y1)

    # # test RMBM
    x = torch.randn(1, 3, 5, 5).cuda()
    rmbm = RMBM(3, 3, with_idt=False).cuda()
    rmbm.eval()
    y0 = rmbm(x)
    RK, RB = rmbm.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print(y0 - y1)
