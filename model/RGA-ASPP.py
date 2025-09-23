import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class AxialAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.in_ch = in_ch
        self.reduction = reduction

        self.q_conv_h = nn.Conv2d(in_ch, in_ch // reduction, 1)
        self.k_conv_h = nn.Conv2d(in_ch, in_ch // reduction, 1)
        self.v_conv_h = nn.Conv2d(in_ch, in_ch, 1)

        self.q_conv_w = nn.Conv2d(in_ch, in_ch // reduction, 1)
        self.k_conv_w = nn.Conv2d(in_ch, in_ch // reduction, 1)
        self.v_conv_w = nn.Conv2d(in_ch, in_ch, 1)

        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        attn_h = F.softmax(self.q_conv_h(x).mean(1, keepdim=True) *
                           self.k_conv_h(x).mean(1, keepdim=True), dim=2)
        out_h = attn_h * self.v_conv_h(x)

        attn_w = F.softmax(self.q_conv_w(x).mean(1, keepdim=True) *
                           self.k_conv_w(x).mean(1, keepdim=True), dim=3)
        out_w = attn_w * self.v_conv_w(x)

        return self.gamma_h * out_h + self.gamma_w * out_w + x

class RGA_ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=64, dilations=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch * 4, 1, bias=False),
            nn.Sigmoid()
        )

        self.dwconv = DepthwiseSeparableConv(in_ch, out_ch)
        self.axial_att = AxialAttention(out_ch, reduction=16)
        self.fuse = DepthwiseSeparableConv(out_ch * 2, in_ch)

    def forward(self, x):
        branch_outs = [self.relu(self.bn(b(x))) for b in self.branches]
        multi_feat = torch.cat(branch_outs, dim=1)
        att = self.channel_att(multi_feat)
        multi_feat = multi_feat * att
        y = sum(torch.chunk(multi_feat, len(self.branches), dim=1))

        x_proj = self.dwconv(x)
        x_proj = self.axial_att(x_proj)

        out = torch.cat([y, x_proj], dim=1)
        out = self.fuse(out)
        return out + x


