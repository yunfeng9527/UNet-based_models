import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution = Depthwise + Pointwise
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                   padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(MobileUNet, self).__init__()

        n1 = 32  # MobileNet 通道数起点比 ResUNet 小
        filters = [n1, n1*2, n1*4, n1*8, n1*16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder (使用 DepthwiseSeparableConv)
        self.conv0_0 = DepthwiseSeparableConv(in_ch, filters[0])
        self.conv1_0 = DepthwiseSeparableConv(filters[0], filters[1])
        self.conv2_0 = DepthwiseSeparableConv(filters[1], filters[2])
        self.conv3_0 = DepthwiseSeparableConv(filters[2], filters[3])
        self.conv4_0 = DepthwiseSeparableConv(filters[3], filters[4])

        # Decoder (cat 拼接 + DepthwiseSeparableConv)
        self.up3 = DepthwiseSeparableConv(filters[4] + filters[3], filters[3])
        self.up2 = DepthwiseSeparableConv(filters[3] + filters[2], filters[2])
        self.up1 = DepthwiseSeparableConv(filters[2] + filters[1], filters[1])
        self.up0 = DepthwiseSeparableConv(filters[1] + filters[0], filters[0])

        # Final conv
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.conv0_0(x)
        x1 = self.conv1_0(self.pool(x0))
        x2 = self.conv2_0(self.pool(x1))
        x3 = self.conv3_0(self.pool(x2))
        x4 = self.conv4_0(self.pool(x3))

        # Decoder
        d3 = torch.cat([x3, self.up(x4)], dim=1)
        d3 = self.up3(d3)

        d2 = torch.cat([x2, self.up(d3)], dim=1)
        d2 = self.up2(d2)

        d1 = torch.cat([x1, self.up(d2)], dim=1)
        d1 = self.up1(d1)

        d0 = torch.cat([x0, self.up(d1)], dim=1)
        d0 = self.up0(d0)

        out = self.final(d0)
        return out



