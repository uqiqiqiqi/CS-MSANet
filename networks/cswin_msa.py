import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size, dim_out=None, num_heads=9, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape

        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


class CARAFE(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=2):
        super().__init__()
        self.adjust_in = nn.Conv2d(dim, ((dim + 3) // 4) * 4, 1) if dim % 4 != 0 else nn.Identity()
        adjusted_dim = ((dim + 3) // 4) * 4

        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(adjusted_dim, adjusted_dim // 4, 1)

        self.encoder = nn.Conv2d(
            adjusted_dim // 4,
            (self.up_factor ** 2) * (self.kernel_size ** 2),
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.out = nn.Conv2d(adjusted_dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.adjust_in(x)

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x


class CARAFE4(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=4):
        super().__init__()
        self.adjust_in = nn.Conv2d(dim, ((dim + 15) // 16) * 16, 1) if dim % 16 != 0 else nn.Identity()
        adjusted_dim = ((dim + 15) // 16) * 16

        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(adjusted_dim, adjusted_dim // 4, 1)

        self.encoder = nn.Conv2d(
            adjusted_dim // 4,
            (self.up_factor ** 2) * (self.kernel_size ** 2),
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.out = nn.Conv2d(adjusted_dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.adjust_in(x)

        kernel_tensor = self.down(x)
        kernel_tensor = self.encoder(kernel_tensor)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x

class DynamicChannelAdjust(nn.Module):
    def __init__(self, in_dim, factor=4):
        super().__init__()
        self.factor = factor
        if in_dim % factor != 0:
            self.adjust = nn.Sequential(
                nn.Conv2d(in_dim, in_dim // 2, 1),
                nn.GELU(),
                nn.Conv2d(in_dim // 2, ((in_dim + factor - 1) // factor) * factor, 1)
            )
        else:
            self.adjust = nn.Identity()

    def forward(self, x):
        return self.adjust(x)


class HybridAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adjusted_dim = ((dim + 3) // 4) * 4

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.adjusted_dim, self.adjusted_dim // 4),
            nn.GELU(),
            nn.Linear(self.adjusted_dim // 4, self.adjusted_dim),
            nn.Sigmoid()
        )

        if dim != self.adjusted_dim:
            self.channel_adjust = nn.Conv2d(dim, self.adjusted_dim, 1)
        else:
            self.channel_adjust = nn.Identity()

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(self.adjusted_dim, self.adjusted_dim, 7, padding=3, groups=8),
            nn.Conv2d(self.adjusted_dim, 1, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.channel_adjust(x)
        b, c, h, w = x.shape

        channel_attn = self.gap(x).view(b, c)
        channel_attn = self.fc(channel_attn).view(b, c, 1, 1)

        spatial_attn = self.spatial_conv(x)

        weights = F.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        return x * (channel_attn * weights[0] + spatial_attn * weights[1])

class SpatialAttentionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.conv(x)
        return x * attn


class MSA(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.adjusted_dim = ((in_dim + 3) // 4) * 4

        self.hybrid_attn = HybridAttention(self.adjusted_dim)

        self.dwc_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.adjusted_dim, self.adjusted_dim, 3, padding=1, groups=self.adjusted_dim),
                nn.BatchNorm2d(self.adjusted_dim),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.Conv2d(self.adjusted_dim, self.adjusted_dim, (1, 3), padding=(0, 1), groups=self.adjusted_dim),
                nn.BatchNorm2d(self.adjusted_dim),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.Conv2d(self.adjusted_dim, self.adjusted_dim, (3, 1), padding=(1, 0), groups=self.adjusted_dim),
                nn.BatchNorm2d(self.adjusted_dim),
                nn.ReLU6()
            )
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(4 * self.adjusted_dim, self.adjusted_dim, 1),
            nn.GroupNorm(min(8, self.adjusted_dim // 4), self.adjusted_dim),
            nn.GELU()
        )

        if self.adjusted_dim != in_dim:
            self.final_conv = nn.Conv2d(self.adjusted_dim, in_dim, 1)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        attn_x = self.hybrid_attn(x)

        branch_outs = [branch(attn_x) for branch in self.dwc_branches]

        concat = torch.cat([attn_x] + branch_outs, dim=1)
        fused = self.fusion(concat)

        fused = fused.reshape(B, C, -1)
        fused = fused.permute(0, 2, 1)

        return fused

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)


class DBA(nn.Module):
    def __init__(self, channels, factor=8):
        super(DBA, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0, "channels must be divisible by groups"
        self.channels = channels

        self.weight_gen = nn.Sequential(
            nn.Conv2d(channels, max(4, channels // 8), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(4, channels // 8)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(max(4, channels // 8), 3, kernel_size=1)
        )

        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels // self.groups, channels // self.groups, 1, groups=channels // self.groups),
            nn.BatchNorm2d(channels // self.groups)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels // self.groups, channels // self.groups, 3, 1, 1, groups=channels // self.groups),
            nn.BatchNorm2d(channels // self.groups)
        )

        self.boundary_enhance = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.LayerNorm([channels // 8, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        dynamic_weights = self.weight_gen(x).sigmoid()  # [b,3,1,1]
        alpha, beta, gamma = dynamic_weights[:, 0], dynamic_weights[:, 1], dynamic_weights[:, 2]

        group_x = x.reshape(b * self.groups, -1, h, w)  # [b*g, c/g, h, w]

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        gca_out = (group_x * weights.sigmoid()).reshape(b, c, h, w)

        boundary_mask = self.boundary_enhance(x)  # [b,1,h,w]
        enhanced_x = x * (1 + gamma.view(-1, 1, 1, 1) * boundary_mask)

        channel_weights = self.channel_recalibrate(x)  # [b,c,1,1]

        out = (alpha.view(-1, 1, 1, 1) * gca_out +
               beta.view(-1, 1, 1, 1) * enhanced_x) * channel_weights

        return out


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=8, embed_dim=64, depth=[1, 2, 9, 1],
                 split_size=[1, 2, 7, 7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        self.stage1_res = img_size // 4
        self.stage2_res = img_size // 8
        self.stage3_res = img_size // 16

        self.ema_skip1 = DBA(embed_dim)
        self.ema_skip2 = DBA(embed_dim * 2)
        self.ema_skip3 = DBA(embed_dim * 4)

        # encoder

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        print("depth", depth)
        self.stage1 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth[0])])
        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.norm = norm_layer(curr_dim)

        # decoder

        self.stage_up4 = nn.ModuleList([MSA(512) for _ in range(depth[-1])])

        self.upsample4 = CARAFE(((curr_dim + 3) // 4) * 4, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear4 = nn.Linear(512, 256)
        self.stage_up3 = nn.ModuleList([MSA(256) for _ in range(depth[2])])

        self.upsample3 = CARAFE(((curr_dim + 3) // 4) * 4, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear3 = nn.Linear(256, 128)
        self.stage_up2 = nn.ModuleList([MSA(128) for _ in range(depth[1])])

        self.upsample2 = CARAFE(((curr_dim + 3) // 4) * 4, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear2 = nn.Linear(128, 64)
        self.stage_up1 = nn.ModuleList([MSA(64) for _ in range(depth[0])])

        self.upsample1 = CARAFE4(((curr_dim + 15) // 16) * 16, 64)
        self.norm_up = norm_layer(embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        # Classifier head

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.stage1_conv_embed(x)

        x = self.pos_drop(x)

        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x_2d = x.permute(0, 2, 1).view(B, C, H, W)
        x1_ema = self.ema_skip1(x_2d)
        self.x1 = x1_ema.view(B, C, -1).permute(0, 2, 1)
        x = self.merge1(x)

        for blk in self.stage2:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x_2d = x.permute(0, 2, 1).view(B, C, H, W)
        x2_ema = self.ema_skip2(x_2d)
        self.x2 = x2_ema.view(B, C, -1).permute(0, 2, 1)
        x = self.merge2(x)

        for blk in self.stage3:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x_2d = x.permute(0, 2, 1).view(B, C, H, W)
        x3_ema = self.ema_skip3(x_2d)
        self.x3 = x3_ema.view(B, C, -1).permute(0, 2, 1)
        x = self.merge3(x)

        for blk in self.stage4:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)

        return x

    # Dencoder and Skip connection
    def forward_up_features(self, x):
        for blk in self.stage_up4:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_up = self.upsample4(x)
        if x_up.size(-1) != self.x3.size(-1):
            x_up = nn.Linear(x_up.size(-1), self.x3.size(-1)).to(x_up.device)(x_up)

        x = torch.cat([self.x3, x_up], dim=-1)
        x = self.concat_linear4(x)

        for blk in self.stage_up3:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_up = self.upsample3(x)
        if x_up.size(-1) != self.x2.size(-1):
            x_up = nn.Linear(x_up.size(-1), self.x2.size(-1)).to(x_up.device)(x_up)

        x = torch.cat([self.x2, x_up], dim=-1)
        x = self.concat_linear3(x)

        for blk in self.stage_up2:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_up = self.upsample2(x)
        if x_up.size(-1) != self.x1.size(-1):
            x_up = nn.Linear(x_up.size(-1), self.x1.size(-1)).to(x_up.device)(x_up)

        x = torch.cat([self.x1, x_up], dim=-1)
        x = self.concat_linear2(x)

        for blk in self.stage_up1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm_up(x)  # B L C
        return x

    def _align_features(self, src, tgt):
        if src.size(-1) != tgt.size(-1):
            device = src.device
            linear = nn.Linear(src.size(-1), tgt.size(-1)).to(device)
            return linear(src)
        return src

    def up_x4(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = self.upsample1(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.output(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.forward_up_features(x)

        x = self.up_x4(x)

        return x
