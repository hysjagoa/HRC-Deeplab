class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# MPSC Block
class MultiScaleParallelStripConvBlock(nn.Module):
    def __init__(self, in_ch):
        super(MultiScaleParallelStripConvBlock, self).__init__()
        # 深度可分离卷积
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)

        # 1×5 + 5×1 分支
        self.branch1_1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 5), padding=(0, 2))
        self.branch1_2 = nn.Conv2d(in_ch, in_ch, kernel_size=(5, 1), padding=(2, 0))
        self.ca1 = ChannelAttention(in_ch)

        # 1×7 + 7×1 分支
        self.branch2_1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_2 = nn.Conv2d(in_ch, in_ch, kernel_size=(7, 1), padding=(3, 0))
        self.ca2 = ChannelAttention(in_ch)

        # 1×11 + 11×1 分支
        self.branch3_1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 11), padding=(0, 5))
        self.branch3_2 = nn.Conv2d(in_ch, in_ch, kernel_size=(11, 1), padding=(5, 0))
        self.ca3 = ChannelAttention(in_ch)

        # 拼接后的1×1卷积压缩
        self.compress = nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        # DWConv
        x_dw = self.dwconv(x)

        # 分支1
        b1 = self.branch1_1(x_dw) + self.branch1_2(x_dw)
        b1 = self.ca1(b1)

        # 分支2
        b2 = self.branch2_1(x_dw) + self.branch2_2(x_dw)
        b2 = self.ca2(b2)

        # 分支3
        b3 = self.branch3_1(x_dw) + self.branch3_2(x_dw)
        b3 = self.ca3(b3)

        # 拼接
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.compress(out)

        # 残差相加
        out = out + x

        return out
