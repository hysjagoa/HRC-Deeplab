import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


#  Stage1 和 Stage3: Dilated Conv + Downsample 
class DilatedConvDown(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

#Stage2: 小波变化下采样 
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        return self.conv_bn_relu(x)


# Stage4: MPSC Block 
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
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


class MultiScaleParallelStripConvBlock(nn.Module):
    def __init__(self, in_ch, stride=2):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1)
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)

        # Branches
        self.branch1_1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1,5), padding=(0,2))
        self.branch1_2 = nn.Conv2d(in_ch, in_ch, kernel_size=(5,1), padding=(2,0))
        self.ca1 = ChannelAttention(in_ch)

        self.branch2_1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1,7), padding=(0,3))
        self.branch2_2 = nn.Conv2d(in_ch, in_ch, kernel_size=(7,1), padding=(3,0))
        self.ca2 = ChannelAttention(in_ch)

        self.branch3_1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1,11), padding=(0,5))
        self.branch3_2 = nn.Conv2d(in_ch, in_ch, kernel_size=(11,1), padding=(5,0))
        self.ca3 = ChannelAttention(in_ch)

        self.compress = nn.Conv2d(in_ch*3, in_ch, kernel_size=1)

    def forward(self, x):
        x = self.down(x)
        x_dw = self.dwconv(x)

        b1 = self.ca1(self.branch1_1(x_dw) + self.branch1_2(x_dw))
        b2 = self.ca2(self.branch2_1(x_dw) + self.branch2_2(x_dw))
        b3 = self.ca3(self.branch3_1(x_dw) + self.branch3_2(x_dw))

        out = torch.cat([b1, b2, b3], dim=1)
        out = self.compress(out)
        return out + x


#Stage5: RGA-ASPP 
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class AxialAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.q_conv_h = nn.Conv2d(in_ch, in_ch//reduction, 1)
        self.k_conv_h = nn.Conv2d(in_ch, in_ch//reduction, 1)
        self.v_conv_h = nn.Conv2d(in_ch, in_ch, 1)

        self.q_conv_w = nn.Conv2d(in_ch, in_ch//reduction, 1)
        self.k_conv_w = nn.Conv2d(in_ch, in_ch//reduction, 1)
        self.v_conv_w = nn.Conv2d(in_ch, in_ch, 1)

        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B,C,H,W = x.shape
        attn_h = F.softmax(self.q_conv_h(x).mean(1, keepdim=True) *
                           self.k_conv_h(x).mean(1, keepdim=True), dim=2)
        out_h = attn_h * self.v_conv_h(x)

        attn_w = F.softmax(self.q_conv_w(x).mean(1, keepdim=True) *
                           self.k_conv_w(x).mean(1, keepdim=True), dim=3)
        out_w = attn_w * self.v_conv_w(x)

        return self.gamma_h*out_h + self.gamma_w*out_w + x


class RGA_ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=64, dilations=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch*4, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch*4, 1, bias=False),
            nn.Sigmoid()
        )

        self.dwconv = DepthwiseSeparableConv(in_ch, out_ch)
        self.axial_att = AxialAttention(out_ch)
        self.fuse = DepthwiseSeparableConv(out_ch*2, in_ch)

    def forward(self, x):
        branch_outs = [self.relu(self.bn(b(x))) for b in self.branches]
        multi_feat = torch.cat(branch_outs, dim=1)
        att = self.channel_att(multi_feat)
        multi_feat = multi_feat * att
        y = sum(torch.chunk(multi_feat, len(self.branches), dim=1))

        x_proj = self.axial_att(self.dwconv(x))
        out = torch.cat([y, x_proj], dim=1)
        out = self.fuse(out)
        return out + x


# our Backbone 
class ResNet18_Modified(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.stage1 = DilatedConvDown(in_ch, base_ch)        # H/2
        self.stage2 = Down_wt(base_ch, base_ch*2)           # H/4
        self.stage3 = DilatedConvDown(base_ch*2, base_ch*4) # H/8
        self.stage4 = MultiScaleParallelStripConvBlock(base_ch*4) # H/16
        self.stage5 = RGA_ASPP(base_ch*4, out_ch=base_ch*8)      # H/16 (no downsample)

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c1, c2, c3, c4, c5



