import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn import functional as F

from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

import math
import numpy as np
import torch

import random
from basicsr.utils import associative_scan, binary_operator_diag
from analysis.utils_fvcore import LRUScan


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class WindowAttention(nn.Module):
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class L_SA(nn.Module):
    # The implementation builds on CAT code https://github.com/zhengchen1999/CAT/blob/main/basicsr/archs/cat_arch.py
    def __init__(self, dim, num_heads,
                 split_size=[2,4], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., idx=0, reso=64, rs_id=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                WindowAttention(
                    dim//2, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)

            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None

            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

    def calculate_mask(self, H, W, device=None):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        if device is None:
            device = torch.device("cpu")  # __init__ 호출 대비
        img_mask_0 = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """

        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)


        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        device = x.device

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            qkv = qkv.view(3, B, _H, _W, C)
            # H-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W, device=device)
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0])
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1])

            else:
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # H-Rwin
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)

        # mix
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + lcm

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LRUcore(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            state_features,
            rmin=0.0,
            rmax=1.0,
            max_phase=6.283,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features
        
        # D parameter
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )

        # Lambda initialization
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        # Gamma initialization
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))
            )
        )
        
        # Complex input projection
        self.B_re = nn.Parameter(torch.randn([state_features, in_features]) / math.sqrt(2 * in_features))
        self.B_im = nn.Parameter(torch.randn([state_features, in_features]) / math.sqrt(2 * in_features))
        
        # Complex output projection
        self.C_re = nn.Parameter(torch.randn([out_features, state_features]) / math.sqrt(state_features))
        self.C_im = nn.Parameter(torch.randn([out_features, state_features]) / math.sqrt(state_features))

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))             # shape (N,)
        lambda_phase = torch.exp(self.theta_log)                    # shape (N,)
        lambda_re = lambda_abs * torch.cos(lambda_phase)            # (N,)
        lambda_im = lambda_abs * torch.sin(lambda_phase)            # (N,)
        lambdas = torch.complex(lambda_re, lambda_im)               # complex (N,)

        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B_re.device)  # shape (N,1)
        B_re_scaled = self.B_re * gammas  # (N, in_features)
        B_im_scaled = self.B_im * gammas
        
        return lambdas, B_re_scaled, B_im_scaled, self.C_re, self.C_im, self.D

    def forward_1d_lru_scan(self, seq_1d, lambdas, B_re, B_im, C_re, C_im, D, promptA, promptB, promptC, state=None):
        B, L, inF = seq_1d.shape
        N = lambdas.shape[0]

        # promptA
        alpha = promptA.mean(dim=0)

        # lambdas
        lam_base = lambdas.view(1, 1, N).expand(B, L, N)
        lam_expand = lam_base * alpha.unsqueeze(0).to(lam_base)

        # Bu
        Bu_re = seq_1d @ B_re.T
        Bu_im = seq_1d @ B_im.T
        Bu_cplx = torch.complex(Bu_re, Bu_im)

        # promptB
        Bu_cplx = Bu_cplx * promptB.to(Bu_cplx)

        if state is not None:
            if state.ndim == 1:
                Bu_cplx[:, 0, :] = Bu_cplx[:, 0, :] + lambdas[None, :] * state[None, :]
            else:
                Bu_cplx[:, 0, :] = Bu_cplx[:, 0, :] + lambdas[None, :] * state

        # scan
        inner_states = associative_scan(binary_operator_diag, (lam_expand, Bu_cplx))[1]

        # state*prompt
        promptC_re, promptC_im = promptC.chunk(2, dim=-1)
        inner_real = inner_states.real*promptC_re
        inner_imag = inner_states.imag*promptC_im

        # projection
        real_part = torch.einsum("bln,on->blo", inner_real, self.C_re)
        imag_part = torch.einsum("bln,on->blo", inner_imag, self.C_im)
       
        # result
        y_lin = real_part - imag_part + seq_1d @ D.T

        return y_lin

    def forward(self, x_1d, promptA, promptB, promptC):
        lambdas, B_re, B_im, C_re, C_im, D = self.ss_params()
        y_lin_1d = self.forward_1d_lru_scan(x_1d, lambdas, B_re, B_im, C_re, C_im, D, promptA, promptB, promptC, state=None)
        return y_lin_1d

    # def forward(self, x_1d, promptA, promptB, promptC):  # For calculating FLOPs
    #     lambdas, B_re, B_im, C_re, C_im, D = self.ss_params()
    #     return LRUScan.apply(x_1d, promptA, promptB, promptC, lambdas, B_re, B_im, C_re, C_im, D)

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def categorize(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class SemanticLRU(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        dropout=0.0,
        rmin=0.0,
        rmax=1.0,
        num_tokens=128,
        hidden_dim=360,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.d_hidden = num_tokens

        # LRU
        self.lru_core = LRUcore(
            in_features=d_model,
            out_features=d_model,
            state_features=d_state,
            rmin=rmin,
            rmax=rmax,
            max_phase=2*math.pi
        )
        self.out_proj = nn.Linear(d_model, d_model // 2, bias=True)

        # Preprocess
        self.in_proj = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, 1, 1, 0),
        )
        self.CPE = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, 3, 1, 1, groups=self.d_model),
        )

        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.wq = nn.Linear(self.d_model, self.d_model // 3, bias=True)
        self.wk = nn.Linear(self.d_model, self.d_model // 3, bias=True)
        self.wv = nn.Linear(self.d_model, self.d_model // 2, bias=True)
        self.scale = nn.Parameter(torch.ones([self.num_tokens]) * 0.5, requires_grad=True)

        # self.gamma_skip = nn.Parameter(1e-1 * torch.ones((self.d_model)), requires_grad=True)

    def forward(self, U, H, W, Learned_dict):

        B, n, C = U.shape

        # QKV generation
        Q_U = self.wq(U)
        K_D = self.wk(Learned_dict)
        V_D = self.wv(Learned_dict)

        # SMU
        SMU = (F.normalize(Q_U, dim=-1) @ F.normalize(K_D, dim=-1).transpose(-2, -1))  # b, n, n_tk
        scale = torch.clamp(self.scale, 0, 1)
        SMU = SMU * (1 + scale * np.log(self.num_tokens))
        Mk = self.softmax(SMU)

        # Mk * V
        Y_enhance = (Mk @ V_D).reshape(B, n, C//2)

        # Mk Chunk
        Mk_AB, Mk_C = Mk.chunk(2, dim = -1)
        Mk_A, Mk_B = Mk_AB.chunk(2, dim = -1)

        # Semantic neighbor sorting
        pred_cls = self.logsoftmax(SMU)    # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_cls, hard=True, dim=-1)  # [B, HW, num_token]
        group_idx = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        _, sort_idx = torch.sort(group_idx, dim=-1, stable=False)
        rev_sort_idx = index_reverse(sort_idx)

        # 2D projection
        U = U.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        U = self.in_proj(U)
        U = U * torch.sigmoid(self.CPE(U))
        cc = U.shape[1]
        U = U.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_U = categorize(U, sort_idx)
        Y_scan = self.lru_core(semantic_U, Mk_A, Mk_B, Mk_C) 
        Y_LRU = categorize(Y_scan, rev_sort_idx)
        Y_LRU = self.out_proj(Y_LRU)

        Y_out = torch.cat([Y_LRU, Y_enhance], dim=-1)

        return Y_out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0, 
                 rs_id=0, split_size=[2,4], shift_size=[1,2], reso=64, layerscale_value=1e-4,
                 d_state=32, rmin=0.9, rmax=0.99, num_tokens=128):
        super().__init__()

        self.idx = idx
        self.rs_id = rs_id
        mlp_hidden_dim = int(dim * mlp_ratio)

        if idx % 2 == 0:
            self.attn = L_SA(
                dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                drop=drop, idx=idx, reso=reso, rs_id=rs_id
            )
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
        else:
            self.lru = SemanticLRU(
                dim, d_state, attn_drop=attn_drop, rmin=rmin, rmax=rmax, num_tokens=num_tokens, hidden_dim=mlp_hidden_dim
            )
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
            self.norm3 = norm_layer(dim)
            self.norm4 = norm_layer(dim)   
            
            self.embedding = nn.Embedding(num_tokens, dim)
            self.embedding.weight.data.uniform_(-1 / num_tokens, 1 / num_tokens)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        res = x

        if self.idx % 2 == 0:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W)) 
        else:
            x = x + self.drop_path(self.lru(self.norm3(x), H, W, self.embedding.weight.repeat(B, 1, 1)))
            x = x + self.drop_path(self.mlp(self.norm4(x), H, W)) 

        x = x + (res * self.gamma)

        return x


class ResidualGroup(nn.Module):

    def __init__(   self,
                    dim,
                    reso,
                    num_heads,
                    mlp_ratio=4.,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_paths=None,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    depth=5,
                    use_chk=False,
                    resi_connection='1conv',
                    rs_id=0,
                    rmin = 0.0,
                    rmax = 1.0,
                    num_tokens=128):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                idx = i,
                rs_id = rs_id,
                split_size = split_size,
                shift_size = [split_size[0]//2, split_size[1]//2],
                rmin = rmin,
                rmax = rmax,
                num_tokens=num_tokens
                )for i in range(depth)])


        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# @ARCH_REGISTRY.register()
class LSM(nn.Module):

    def __init__(self,
                img_size=64,
                in_chans=3,
                embed_dim=180,
                depth=[10,10,10,10],
                num_heads=[4,4,4,4],
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_chk=False,
                upscale=2,
                img_range=1.,
                resi_connection='1conv',
                split_size=[8,8],
                rmin=0.9,
                rmax=0.99,
                num_tokens=128,
                **kwargs):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rs_id=i,
                split_size=split_size,
                rmin=rmin,
                rmax=rmax,
                num_tokens=num_tokens
                )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, Reconstruction ------------------------- #
        # for classical SR
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        """

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # for classical SR
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x


def buildLSMS(upscale=2):
    return LSM(
            upscale=upscale,
            img_size=128,
            in_chans=3,
            img_range=1.,
            d_state=32,
            depth=[5,5,5,5,5,5],
            embed_dim=180,
            num_heads=[5,5,5,5,5,5],
            mlp_ratio=2,
            resi_connection='1conv',
            split_size=[8, 32],
            c_ratio=0.5,
            rmin=0.9,
            rmax=0.99,
            inner_rank=64,
            num_tokens=128
            )

def buildLSM(upscale=2):
    return LSM(
            upscale=upscale,
            img_size=128,
            in_chans=3,
            img_range=1.,
            d_state=32,
            depth=[5,5,5,5,5,5,5,5],
            embed_dim=180,
            num_heads=[5,5,5,5,5,5,5,5],
            mlp_ratio=2,
            resi_connection='1conv',
            split_size=[8, 32],
            c_ratio=0.5,
            rmin=0.9,
            rmax=0.99,
            inner_rank=64,
            num_tokens=128
            )
