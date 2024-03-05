import math
from sympy import true
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.layers import CALayer
from scripts.utils import pad_tensor, pad_tensor_back
import pdb
import numbers
from einops import rearrange

def exists(x):
    return x is not None

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class TLU(nn.Module):
    def __init__(self, dim, tlu_factor, bias):
        super(TLU, self).__init__()

        hidden_features = int(dim * tlu_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32, use_CA=False):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.use_CA = use_CA
        if self.use_CA:
            self.ca_block = CALayer(dim_out)

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        if time_emb is not None:
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        if self.use_CA:
            h = self.ca_block(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class TLB(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, \
        with_attn=False, use_affine_level=False, use_CA=False, attn_type=None,use_tlu=True):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)
        self.with_attn = with_attn
        self.use_tlu = use_tlu
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, use_affine_level=use_affine_level, use_CA=use_CA)
        if use_tlu is True :
            self.tlu = TLU(dim_out, tlu_factor=4.0, bias=False)
            self.norm = LayerNorm(dim_out, LayerNorm_type='WithBias')
            self.conv_du = nn.Sequential(
            nn.Linear(dim_out,dim_out // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out // 4, dim_out, bias=False),
            nn.Sigmoid()
                )
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)
    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.use_tlu:
            h = self.noise_func(x, time_emb)
            x = x + self.tlu(self.norm(h))
        if(self.with_attn):
            x = self.attn(x)
        return x

@ARCH_REGISTRY.register()
class TLBNet(nn.Module):
    def __init__(
        self,
        in_channel=13,
        out_channel=3,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4),
        attn_res=(16),
        res_blocks=2,
        dropout=0.2,
        with_noise_level_emb=True,
        image_size=128,
        use_affine_level=False,
        use_CA=False,
        attn_type=None,
        divide=None,
        use_tlu=True,
        drop2d_input=False,
        drop2d_input_p=0.0,
        channel_randperm_input=False
    ):
        super().__init__()
        self.drop2d_input = drop2d_input
        if self.drop2d_input:
            self.drop2d_in = nn.Dropout2d(drop2d_input_p)
        
        self.channel_randperm_input = channel_randperm_input

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        
        self.divide = divide

        num_mults = len(channel_mults) 
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(TLB( 
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, \
                    dropout=dropout, with_attn=use_attn, use_affine_level=use_affine_level, \
                        use_CA=use_CA, attn_type=attn_type,use_tlu=use_tlu))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2  
        self.downs = nn.ModuleList(downs)
        mid_use_attn =(len(attn_res) != 0)
        self.mid = nn.ModuleList([
            TLB(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=mid_use_attn, use_affine_level=use_affine_level, use_CA=use_CA,
                               attn_type=attn_type,use_tlu=use_tlu),
            TLB(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, use_affine_level=use_affine_level, use_CA=use_CA,
                               attn_type=attn_type,use_tlu=use_tlu)
        ]) 

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(TLB( 
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn, use_affine_level=use_affine_level, use_CA=use_CA, attn_type=attn_type,use_tlu=use_tlu))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel)) 
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
    def forward(self, x, time=None):
        if self.channel_randperm_input:
            from scripts.pytorch_utils import channel_randperm
            x[:, :6, ...] = channel_randperm(x[:, :6, ...])
        if self.drop2d_input:
            x[:, :6, ...] = self.drop2d_in(x[:, :6, ...])
        if self.divide:
            x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x, self.divide)
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, TLB):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, TLB):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, TLB):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        if self.divide:
            out = self.final_conv(x)
            out = pad_tensor_back(out, pad_left, pad_right, pad_top, pad_bottom)
            return out
        else:
            return self.final_conv(x)
