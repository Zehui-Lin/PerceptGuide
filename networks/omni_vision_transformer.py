import copy
import logging

import math
import torch
import torch.nn as nn

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torch.functional import F


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ChannelHalf(nn.Module):
    def __init__(self, input_resolution=None, dim=0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear = nn.Linear(dim, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)
        self.input_resolution = input_resolution

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim  # reduction 4 * self.dim -> 2 * self.dim
        return flops


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


logger = logging.getLogger(__name__)

# POSITION_LEN = 10
# TASK_LEN = 2
# MODE_LEN = 3
# TYPE_LEN = 2

from datasets.omni_dataset import position_prompt_one_hot_dict
from datasets.omni_dataset import task_prompt_one_hot_dict
from datasets.omni_dataset import mode_prompt_one_hot_dict
from datasets.omni_dataset import type_prompt_one_hot_dict

POSITION_LEN = len(position_prompt_one_hot_dict)
TASK_LEN = len(task_prompt_one_hot_dict)
MODE_LEN = len(mode_prompt_one_hot_dict)
TYPE_LEN = len(type_prompt_one_hot_dict)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        prompt=False,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.prompt = prompt

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if prompt:

            self.hyper_param_position = nn.Parameter(torch.rand(1, num_heads, dim // num_heads, 1))
            self.hyper_param_task = nn.Parameter(torch.rand(1, num_heads, dim // num_heads, 1))
            self.hyper_param_mode = nn.Parameter(torch.rand(1, num_heads, dim // num_heads, 1))
            self.hyper_param_type = nn.Parameter(torch.rand(1, num_heads, dim // num_heads, 1))

            trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if self.prompt:
            x, position_embed, task_embed, mode_embed, type_embed = x
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        if self.prompt:
            position_embed = position_embed.reshape(-1, self.num_heads, 1, C // self.num_heads)
            task_embed = task_embed.reshape(-1, self.num_heads, 1, C // self.num_heads)
            mode_embed = mode_embed.reshape(-1, self.num_heads, 1, C // self.num_heads)
            type_embed = type_embed.reshape(-1, self.num_heads, 1, C // self.num_heads)

            W_p = self.hyper_param_position * position_embed
            W_t = self.hyper_param_task * task_embed
            W_m = self.hyper_param_mode * mode_embed
            W_ty = self.hyper_param_type * type_embed

            W_all = W_p * W_t * W_m * W_ty

            B = W_all.shape[0]
            B_ = x.shape[0]
            num_windows = B_ // B
            W_all = W_all.repeat(num_windows, 1, 1, 1)
            attn = q @ W_all @ k.transpose(-2, -1)
        else:
            attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        prompt=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.prompt = prompt
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if prompt:
            self.position_prompt_proj_mlp = nn.Linear(POSITION_LEN, dim)
            self.task_prompt_proj_mlp = nn.Linear(TASK_LEN, dim)
            self.mode_prompt_proj_mlp = nn.Linear(MODE_LEN, dim)
            self.type_prompt_proj_mlp = nn.Linear(TYPE_LEN, dim)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            prompt=prompt,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        if self.prompt:
            x, position_prompt, task_prompt, mode_prompt, type_prompt = x
            position_embed = self.position_prompt_proj_mlp(position_prompt)
            task_embed = self.task_prompt_proj_mlp(task_prompt)
            mode_embed = self.mode_prompt_proj_mlp(mode_prompt)
            type_embed = self.type_prompt_proj_mlp(type_prompt)

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.prompt:
            attn_windows = self.attn(
                (x_windows, position_embed, task_embed, mode_embed, type_embed),
                mask=self.attn_mask,
            )
        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        res_scale=None,
        use_checkpoint=False,
        prompt=False,
        this_layer_prompt=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.prompt = prompt
        self.num_heads = num_heads
        self.this_layer_prompt = this_layer_prompt

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    prompt=prompt and this_layer_prompt,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if res_scale is not None:
            self.res_scale = res_scale(input_resolution, dim)
        else:
            self.res_scale = None

    def forward(self, x):
        if self.prompt:
            x, position_prompt, task_prompt, mode_prompt, type_prompt = x

        for blk in self.blocks:
            if self.prompt and self.this_layer_prompt:
                if self.use_checkpoint:
                    if self.prompt:
                        x = checkpoint.checkpoint(
                            blk,
                            (x, position_prompt, task_prompt, mode_prompt, type_prompt),
                        )
                else:
                    if self.prompt:
                        x = blk((x, position_prompt, task_prompt, mode_prompt, type_prompt))
            else:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

        if self.res_scale is not None:
            x = self.res_scale(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        encoder_depths=[2, 2, 2, 2],
        decoder_depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        ape=False,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        prompt=False,
    ):
        super().__init__()

        print("SwinTransformer architecture information:")

        self.num_layers = len(encoder_depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.prompt = prompt
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))]

        ## Encoder + bottleneck ##
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            this_layer_prompt = False

            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=encoder_depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=enc_dpr[sum(encoder_depths[:i_layer]) : sum(encoder_depths[: i_layer + 1])],
                norm_layer=norm_layer,
                res_scale=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                prompt=prompt,
                this_layer_prompt=this_layer_prompt,
            )
            self.layers.append(layer)

        ## Multi Decoder ##

        self.layers_task_seg_up = nn.ModuleList()
        self.layers_task_seg_skip = nn.ModuleList()
        self.layers_task_seg_head = nn.ModuleList()

        self.layers_task_cls_up = nn.ModuleList()
        self.layers_task_cls_skip = nn.ModuleList()
        self.layers_task_cls_head = nn.ModuleList()

        # stochastic depth
        dec_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]

        for i_layer in range(self.num_layers):

            if prompt:
                this_layer_prompt = True
            else:
                this_layer_prompt = False

            # seg
            self.layers_task_seg_skip.append(
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )  # concat back linear
            if i_layer == 0:
                self.layers_task_seg_up.append(
                    PatchExpand(
                        input_resolution=(
                            patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                            patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                        ),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                        dim_scale=2,
                        norm_layer=norm_layer,
                    )
                )
            else:
                self.layers_task_seg_up.append(
                    BasicLayer(
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                        input_resolution=(
                            patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                            patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                        ),
                        depth=decoder_depths[(self.num_layers - 1 - i_layer)],
                        num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[
                            sum(decoder_depths[: (self.num_layers - 1 - i_layer)]) : sum(
                                decoder_depths[: (self.num_layers - 1 - i_layer) + 1]
                            )
                        ],
                        norm_layer=norm_layer,
                        res_scale=(PatchExpand if (i_layer < self.num_layers - 1) else None),
                        use_checkpoint=use_checkpoint,
                        prompt=prompt,
                        this_layer_prompt=this_layer_prompt,
                    )
                )  # downsample
            # cls
            self.layers_task_cls_skip.append(
                nn.Linear(
                    3 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )  # concat back linear
            if i_layer == 0:
                pass
            else:
                self.layers_task_cls_up.append(
                    BasicLayer(
                        dim=int(embed_dim * 2 ** (self.num_layers - i_layer)),
                        input_resolution=(
                            patches_resolution[0] // (2 ** (self.num_layers - 1 - 0)),
                            patches_resolution[1] // (2 ** (self.num_layers - 1 - 0)),
                        ),
                        depth=decoder_depths[(self.num_layers - i_layer)],
                        num_heads=num_heads[(self.num_layers - i_layer)],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dec_dpr[
                            sum(decoder_depths[: (self.num_layers - i_layer)]) : sum(
                                decoder_depths[: (self.num_layers - i_layer) + 1]
                            )
                        ],
                        norm_layer=norm_layer,
                        res_scale=(ChannelHalf if (i_layer < self.num_layers - 1) else None),
                        use_checkpoint=use_checkpoint,
                        prompt=prompt,
                        this_layer_prompt=this_layer_prompt,
                    )
                )

        self.layers_task_seg_head.append(
            FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim=embed_dim,
            )
        )
        self.layers_task_seg_head.append(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=2,
                kernel_size=1,
                bias=False,
            )
        )

        self.layers_task_cls_head.append(nn.Linear(self.embed_dim * 2, 2))

        ## Norm Layer ##
        self.norm = norm_layer(self.num_features)
        self.norm_task_seg = norm_layer(self.embed_dim)
        self.norm_task_cls = norm_layer(self.embed_dim * 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    # Encoder and Bottleneck
    def forward_features(self, x):
        if self.prompt:
            x, position_prompt, task_prompt, mode_prompt, type_prompt = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            if self.prompt:
                x = layer((x, position_prompt, task_prompt, mode_prompt, type_prompt))
            else:
                x = layer(x)

        x = self.norm(x)

        return x, x_downsample

    # Decoder task head
    def forward_task_features(self, x, x_downsample):
        if self.prompt:
            x, position_prompt, task_prompt, mode_prompt, type_prompt = x
        # seg
        for inx, layer_seg in enumerate(self.layers_task_seg_up):
            if inx == 0:
                x_seg = layer_seg(x)
            else:
                x_seg = torch.cat([x_seg, x_downsample[3 - inx]], -1)
                x_seg = self.layers_task_seg_skip[inx](x_seg)
                if self.prompt:
                    x_seg = layer_seg((x_seg, position_prompt, task_prompt, mode_prompt, type_prompt))
                else:
                    x_seg = layer_seg(x_seg)

        x_seg = self.norm_task_seg(x_seg)

        H, W = self.patches_resolution
        B, _, _ = x_seg.shape
        x_seg = self.layers_task_seg_head[0](x_seg)
        x_seg = x_seg.view(B, 4 * H, 4 * W, -1)
        x_seg = x_seg.permute(0, 3, 1, 2)
        x_seg = self.layers_task_seg_head[1](x_seg)

        # cls
        for inx, layer_cls in enumerate(self.layers_task_cls_up):
            [B, L, C] = x_downsample[2 - inx].shape
            l = int(math.sqrt(L))
            if inx == 0:
                x_cls = x
                x_cls_hw = l // 2
            x_downsample_inx = x_downsample[2 - inx].view(B, l, l, C)
            x_downsample_inx = F.adaptive_avg_pool2d(x_downsample_inx.permute((0, 3, 1, 2)), (x_cls_hw, x_cls_hw))
            x_downsample_inx = x_downsample_inx.permute((0, 2, 3, 1)).view(B, -1, C).contiguous()
            x_cls = torch.cat([x_cls, x_downsample_inx], -1)
            x_cls = self.layers_task_cls_skip[inx + 1](x_cls)
            if inx == 0:
                x = x_cls
            if inx == 0:
                if self.prompt:
                    x_cls = layer_cls((x, position_prompt, task_prompt, mode_prompt, type_prompt))
                else:
                    x_cls = layer_cls(x)
            else:
                if self.prompt:
                    x_cls = layer_cls((x_cls, position_prompt, task_prompt, mode_prompt, type_prompt))
                else:
                    x_cls = layer_cls(x_cls)

        x_cls = self.norm_task_cls(x_cls)

        B, _, _ = x_cls.shape
        x_cls = x_cls.transpose(1, 2)
        x_cls = F.adaptive_avg_pool1d(x_cls, 1).view(B, -1)

        x_cls = self.layers_task_cls_head[0](x_cls)

        return (x_seg, x_cls)

    def forward(self, x):
        if self.prompt:
            _, position_prompt, task_prompt, mode_prompt, type_prompt = x
            x, x_downsample = self.forward_features(x)
            x_tuple = self.forward_task_features(
                (x, position_prompt, task_prompt, mode_prompt, type_prompt),
                x_downsample,
            )
        else:
            x, x_downsample = self.forward_features(x)
            x_tuple = self.forward_task_features(x, x_downsample)
        return x_tuple


class OmniVisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        prompt=False,
    ):
        super(OmniVisionTransformer, self).__init__()
        self.config = config
        self.prompt = prompt

        self.swin = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            encoder_depths=[i + 2 for i in config.MODEL.SWIN.DEPTHS],
            decoder_depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            prompt=prompt,
        )

    def forward(self, x):
        if self.prompt:
            image = x[0].squeeze(1).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            position_prompt = x[1]
            task_prompt = x[2]
            mode_prompt = x[3]
            type_prompt = x[4]
            result = self.swin((image, position_prompt, task_prompt, mode_prompt, type_prompt))
        else:
            x = x.squeeze(1).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            result = self.swin(x)
        return result

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            pretrained_dict = pretrained_dict["model"]
            print("---start load pretrained model of swin encoder---")
            model_dict = self.swin.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")

    def load_from_self(self, pretrained_path):
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "module.swin." in k:
                current_k = k[12:]
                full_dict.update({current_k: v})
                del full_dict[k]

        self.swin.load_state_dict(full_dict, strict=False)
