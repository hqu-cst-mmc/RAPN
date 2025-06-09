import torch
import torch.nn as nn
import math
import warnings
from timm.models.layers import DropPath, trunc_normal_
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, out_dim,segment_frame_num, sub_video_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, v):
        # v:[bs,3,24,512]
        # bs,3,24,512
        B, N, T, dim = v.shape
        # v:[bs*3,25,512]
        q = self.q_map(v).view(B*N,T,self.num_heads,dim // self.num_heads).permute(0,2,1,3)
        k = self.k_map(v).view(B*N,T,self.num_heads,dim // self.num_heads).permute(0,2,1,3)
        v = self.v_map(v).view(B*N,T,self.num_heads,dim // self.num_heads).permute(0,2,1,3)

        # attn=[bs*3,8,24,8]@[bs*3,8,8,24]=[bs*3,8,24,24]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #    x=[bs,8,5,80]@[bs,8,80,8]=[bs,8,5,8]->[bs,5,64]
        x = (attn @ v).transpose(1, 2).reshape(B, T, dim).unsqueeze(1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, segment_frame_num, sub_video_num, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = SelfAttention(
            dim, dim,segment_frame_num, sub_video_num, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    # q是视频1 step_i的特征,v是视频2 step_i的特征图



    def forward(self, v):
        # 自注意力
        v = v + self.drop_path(self.attn(self.norm_v(v)))
        v = v + self.drop_path(self.mlp(self.norm2(v)))
        return v


class encoder_fuser(nn.Module):
    def __init__(self, dim, num_heads, num_layers,segment_frame_num,segments_num,allframes, MSA_num=3):
        super(encoder_fuser, self).__init__()
        model_list = []
        self.encoder_layers = num_layers
        for i in range(num_layers):
            model_list.append(DecoderBlock(dim, num_heads, segment_frame_num, segments_num))
        self.model = nn.ModuleList(model_list)

        # MixSelf-Attention
        MSA_list = []
        self.MSA_num = MSA_num
        for i in range(self.MSA_num):
            MSA_list.append(DecoderBlock(dim, num_heads, segment_frame_num, segments_num))
        self.MSA = nn.ModuleList(MSA_list)

        # cls_tokens
        self.token_g = nn.Parameter(torch.zeros(1, 1, dim))
        self.token_l = nn.Parameter(torch.zeros(1, 1, dim))
        # pos_embedding
        self.pos_embed_g = nn.Parameter(torch.zeros(1, allframes + 1, dim))
        self.pos_embed_l = nn.Parameter(torch.zeros(1, allframes + 1, dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # 正态分布取值
        def _no_grad_trunc_normal_(tensor, mean, std, a, b):
            def norm_cdf(x):
                # Computes standard normal cumulative distribution function
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                              "The distribution of values may be incorrect.",
                              stacklevel=2)

            with torch.no_grad():
                # Values are generated by using a truncated uniform distribution and
                # then using the inverse CDF for the normal distribution.
                # Get upper and lower cdf values
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)

                # Uniformly fill tensor with values from [l, u], then translate to
                # [2l-1, 2u-1].
                tensor.uniform_(2 * l - 1, 2 * u - 1)

                # Use inverse cdf transform for normal distribution to get truncated
                # standard normal
                tensor.erfinv_()

                # Transform to proper mean, std
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)

                # Clamp to ensure it's in the proper range
                tensor.clamp_(min=a, max=b)
                return tensor

        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            # type: (Tensor, float, float, float, float) -> Tensor
            return _no_grad_trunc_normal_(tensor, mean, std, a, b)

        trunc_normal_(self.token_g, std=.02)
        trunc_normal_(self.token_l, std=.02)
        trunc_normal_(self.pos_embed_g, std=.02)
        trunc_normal_(self.pos_embed_l, std=.02)

    # f_g是全局视频特征，f_l是局部视频特征
    def forward(self, f_g, f_l):
        # 对应两个视频（g和l）的帧级特征，每个视频共64帧，特征长度768
        # f_g:[bs, 1, 64, 768], f_l:[bs, 1, 64, 768]
        B, N, T, dim = f_g.shape  # 1,3,24,512
        f_g = f_g.view(B * N, T, dim)
        f_l = f_l.view(B * N, T, dim)

        # cls_tokens - 加入cls_tokens
        token_g = self.token_g.expand(B * N, -1, -1)
        token_l = self.token_l.expand(B * N, -1, -1)
        # f_g:[bs, 1, 65, 768], f_l:[bs, 1, 65, 768]
        f_g = torch.cat((token_g, f_g), dim=1)
        f_l = torch.cat((token_l, f_l), dim=1)
        # cls_tokens

        # pos_embbed
        # ----- g -----
        # f_g = f_g + self.pos_embed_g
        f_g = self.pos_drop(f_g)
        f_g = f_g.view(B, N, T+1, dim)
        # ----- l -----
        # f_l = f_l + self.pos_embed_l
        f_l = self.pos_drop(f_l)
        f_l = f_l.view(B, N, T+1, dim)
        # pos_embbed

        # 保存中间变量
        mixseq_list = [torch.zeros_like(f_g) for _ in range(self.MSA_num)]
        # MSA[0] 将视频g的cls_tokens和视频l的帧级特征序列列表组合形成新的特征序列列表送入self.MSA进行自注意力运算。在注意力运算后将mixseq_list的cls_tokens赋值给原来f_g的cls_tokens以更新
        mixseq_list[0][:, :, 0, :] = f_g[:, :, 0, :].clone()
        mixseq_list[0][:, :, 1:, :] = f_l[:, :, 1:, :].detach().clone()
        mixseq_list[0] = self.MSA[0](mixseq_list[0])
        f_g[:, :, 0, :] = mixseq_list[0][:, :, 0, :].clone()
        for it_layer, _layer in enumerate(self.model):
            f_g = _layer(f_g)
            f_l = _layer(f_l)
            # MSA[1]
            if it_layer == (self.encoder_layers // 2):
                mixseq_list[1][:, :, 0, :] = f_g[:, :, 0, :].clone()
                mixseq_list[1][:, :, 1:, :] = f_l[:, :, 1:, :].detach().clone()
                mixseq_list[1] = self.MSA[1](mixseq_list[1])
                f_g[:, :, 0, :] = mixseq_list[1][:, :, 0, :].clone()
        # MSA[2]
        mixseq_list[2][:, :, 0, :] = f_g[:, :, 0, :].clone()
        mixseq_list[2][:, :, 1:, :] = f_l[:, :, 1:, :].detach().clone()
        mixseq_list[2] = self.MSA[2](mixseq_list[2])
        f_g[:, :, 0, :] = mixseq_list[2][:, :, 0, :].clone()
        return f_g, f_l
        # return v
