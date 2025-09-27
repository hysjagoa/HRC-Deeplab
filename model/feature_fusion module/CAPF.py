import torch
import torch.nn as nn
import torch.nn.functional as F

//加入骨干时需要做特征图尺寸以及通道对齐
class CrossAttention(nn.Module):
    def __init__(self, in_ch, heads=4, dim_head=32):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(in_ch, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(in_ch, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(in_ch, inner_dim, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, in_ch, 1),
            nn.BatchNorm2d(in_ch)
        )

    def forward(self, x1, x2):
        """
        Cross attention between x1 and x2
        x1: (B,C,H,W)
        x2: (B,C,H,W)
        """
        b, c, h, w = x1.shape
        q = self.to_q(x1).view(b, self.heads, -1, h*w)   # (B, heads, dim, N)
        k = self.to_k(x2).view(b, self.heads, -1, h*w)
        v = self.to_v(x2).view(b, self.heads, -1, h*w)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.view(b, -1, h, w)
        return self.to_out(out)
    

class CAPF_Fusion(nn.Module):
    def __init__(self, c1_ch, c2_ch, c3_ch, out_ch=128):
        super().__init__()
        # 1x1 conv to align channels
        self.conv1 = nn.Conv2d(c1_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(c2_ch, out_ch, 1)
        self.conv3 = nn.Conv2d(c3_ch, out_ch, 1)

        self.cross_att1 = CrossAttention(out_ch)  # f1 <-> f2
        self.cross_att2 = CrossAttention(out_ch)  # f3 <-> a

        self.conv_after = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, c1, c2, c3):
        f1 = self.conv1(c1)
        f2 = self.conv2(c2)
        a = self.cross_att1(f1, f2)   # 第一层 cross attention

        f3 = self.conv3(c3)
        b = self.cross_att2(f3, a)    # 第二层 cross attention
        c = self.conv_after(b)        # 卷积

        out = torch.cat([c, a], dim=1)  # concat
        return out
