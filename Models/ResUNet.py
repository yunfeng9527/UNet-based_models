import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block for ResUNet
    Conv -> BN -> ReLU -> Conv -> BN
    with skip connection
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # 如果输入通道数和输出通道数不一致，用 1x1 conv 对齐
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ResUNet, self).__init__()

        n1 = 64
        filters = [n1, n1*2, n1*4, n1*8, n1*16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = ResidualBlock(in_ch, filters[0])
        self.conv1_0 = ResidualBlock(filters[0], filters[1])
        self.conv2_0 = ResidualBlock(filters[1], filters[2])
        self.conv3_0 = ResidualBlock(filters[2], filters[3])
        self.conv4_0 = ResidualBlock(filters[3], filters[4])

        # Decoder
        self.up3 = ResidualBlock(filters[4] + filters[3], filters[3])
        self.up2 = ResidualBlock(filters[3] + filters[2], filters[2])
        self.up1 = ResidualBlock(filters[2] + filters[1], filters[1])
        self.up0 = ResidualBlock(filters[1] + filters[0], filters[0])

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
        d3 = self.up(torch.cat([x3, self.up(x4)], dim=1))
        d3 = self.up3(d3)

        d2 = self.up(torch.cat([x2, self.up(d3)], dim=1))
        d2 = self.up2(d2)

        d1 = self.up(torch.cat([x1, self.up(d2)], dim=1))
        d1 = self.up1(d1)

        d0 = self.up(torch.cat([x0, self.up(d1)], dim=1))
        d0 = self.up0(d0)

        out = self.final(d0)
        return out



