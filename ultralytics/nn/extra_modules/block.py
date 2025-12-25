import torch
import torch.nn as nn
from einops import rearrange
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ['EMAC', 'C3_CFEMA', 'CFEMA', 'SDFM']


class EMAC(nn.Module):
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        _, c, _, _ = x.size()
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)

        return x


class EMAB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = EMAC(c2)


class C3_CFEMA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(EMAB(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


class CFEMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EMAB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class SDFM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(SDFM, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(2 * channels, 2 * inter_channels),
            Conv(2 * inter_channels, 2 * channels, act=nn.Sigmoid()),
        )

        self.channel_agg = Conv(2 * channels, channels)

        self.local_att = nn.Sequential(
            Conv(channels, inter_channels, 1),
            Conv(inter_channels, channels, 1, act=False),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(channels, inter_channels, 1),
            Conv(inter_channels, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x1, x2 = data
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input)
        local_w = self.local_att(agg_input)
        global_w = self.global_att(agg_input)
        w = self.sigmoid(local_w * global_w)
        xo = w * x1 + (1 - w) * x2
        return xo
