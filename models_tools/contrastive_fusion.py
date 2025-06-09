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


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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

    def forward(self, q, v):
        # v:[bs,N,5,768]
        # q:[bs,N,1,768]
        B, q_T, dim = q.shape
        # k,v
        k = v
        v_T = v.size(-2)
        # v:[bs,N,5,12,64] q:[bs,N,1,12,64]
        # 视频级特征做kv
        q = self.q_map(q).view(B, q_T, self.num_heads, dim // self.num_heads)
        k = self.k_map(k).view(B, v_T, self.num_heads, dim // self.num_heads)
        v = self.v_map(v).view(B, v_T, self.num_heads, dim // self.num_heads)
        # 帧级特征做kv
        # q = self.q_map(q).view(B, N, q_T, self.num_heads, dim // self.num_heads).permute(0, 1, 3, 2, 4)
        # k = self.k_map(k).view(B, N, v_T, self.num_heads, dim // self.num_heads).permute(0, 1, 3, 2, 4)
        # v = self.v_map(v).view(B, N, v_T, self.num_heads, dim // self.num_heads).permute(0, 1, 3, 2, 4)
        # attn=[bs,N,1,12,64]@[bs,N,5,64,12]=[bs,N,5,12,12]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x =  [bs,N,5,12,12]@[bs,N,5,12,64]=[bs,N,5,12,64]
        # x =  [bs,N,5,12,64]->[bs,N,5,768]
        x = (attn @ v).reshape(B, q_T, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        # 裁判内部特征做自注意力
        q = self.q_map(v).view(B*N, T, self.num_heads, dim // self.num_heads)
        k = self.k_map(v).view(B*N, T, self.num_heads, dim // self.num_heads)
        v = self.v_map(v).view(B*N, T, self.num_heads, dim // self.num_heads)
        # 裁判之间做自注意力
        # q = self.q_map(v).view(B*N,T,self.num_heads,dim // self.num_heads).permute(0,2,1,3)
        # k = self.k_map(v).view(B*N,T,self.num_heads,dim // self.num_heads).permute(0,2,1,3)
        # v = self.v_map(v).view(B*N,T,self.num_heads,dim // self.num_heads).permute(0,2,1,3)

        # attn=[bs*3,8,24,8]@[bs*3,8,8,24]=[bs*3,8,24,24]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #    x=[bs,8,5,80]@[bs,8,80,8]=[bs,8,5,8]->[bs,5,64]
        # x = (attn @ v).transpose(1, 2).reshape(B, N, T, dim)
        x = (attn @ v).reshape(B, N, T, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    # q是视频1 step_i的特征,v是视频2 step_i的特征图
    def forward(self, q, v):
        v = v + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        v = v + self.drop_path(self.mlp(self.norm2(v)))
        return v

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = SelfAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
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

class decoder_fuser(nn.Module):
    def __init__(self, dim, num_heads, num_layers, query_num=7):
        super(decoder_fuser, self).__init__()
        model_list = []
        selfmodel_list = []
        for i in range(num_layers):
            model_list.append(DecoderBlock(dim, num_heads,drop_path=0.1,drop=0.1))
        self.model = nn.ModuleList(model_list)
        self.selfmodel = nn.MultiheadAttention(dim, num_heads=8, dropout=0.1)
        # for i in range(num_layers):
        #     selfmodel_list.append(EncoderBlock(dim, num_heads))
        # self.selfmodel = nn.ModuleList(selfmodel_list)
        # querys
        self.querys = nn.Parameter(torch.zeros(1, query_num, dim))
        # pos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, query_num, dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.att_drop = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)
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
        # 初始化参数
        trunc_normal_(self.querys, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    # q是视频1 step_i的特征,v是视频2 step_i的特征图
    def forward(self, q, v):
        # self.querys->1,5,768
        # train->bs,1,48,768
        # test ->bs,3,48,768
        # B, V, T, dim = v.shape
        # train->bs,1,48,768
        # test ->bs,3,48,768
        # value->bs,V,48,768
        # query->bs,V,5,768

        # self.querys->1,5,768
        # train->bs,1,1,768
        # test ->bs,3,1,768
        B, T, dim = v.shape

        # 交叉注意力
        for _layer in self.model:
            v = _layer(q, v)
        # # 自注意力
        # q = q.squeeze(1)
        # pos_q = q + self.pos_embed
        # q = self.selfmodel(pos_q.permute(1,0,2).contiguous(),pos_q.permute(1,0,2).contiguous(),q.permute(1,0,2).contiguous())[0]
        # q = q.permute(1,0,2).contiguous()
        # q = q + self.att_drop(q)
        # q = self.att_drop(self.mlp(self.norm(q)))
        # # # 自注意力
        # for _layer in self.selfmodel:
        #     q = _layer(q)

        # for self_layer,cross_layer in zip(self.selfmodel,self.model):
        #     q = cross_layer(q, v)
        #     q = self_layer(q)
        return v
