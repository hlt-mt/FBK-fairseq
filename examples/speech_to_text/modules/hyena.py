# Copyright 2024 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fftconv_ref(u, k, D, dropout=None, gelu=True, causal=True):
    # FFT-based convolution with residual connection
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')
    if causal:
        y = y[..., :seqlen]
    else:
        to_skip = seqlen // 2
        y = y[..., to_skip:to_skip + seqlen]
    if dropout is not None:
        y = dropout(y)
    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class ComplexExponentialPositionalEmbedding(nn.Module):
    """Complex exponential positional embeddings for Hyena filters."""
    def __init__(self, emb_dim: int, seq_len: int):
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filters is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len).unsqueeze(1).unsqueeze(0)  # 1, L, 1

        assert emb_dim > 1
        bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len).unsqueeze(1).unsqueeze(0)  # 1, L, 1
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands).unsqueeze(0).unsqueeze(0)
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register_buffer("z", z, persistent=False)
        self.register_buffer("t", t, persistent=False)

    def forward(self, input_len):
        return self.z[:, :input_len], self.t[:, :input_len]


class ExponentialModulation(nn.Module):
    def __init__(
            self,
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=1e-2,
            shift: float = 0.0):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(
            min_decay, max_decay, d_model, requires_grad=False).unsqueeze(0).unsqueeze(0)
        self.register_buffer("deltas", deltas, persistent=False)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
            self,
            d_model,
            emb_dim=3,
            order=16,
            fused_fft_conv=False,
            modulate=True,
            seq_len=1024,
            dropout=0.0,
            w=1,
            bias=True,
            num_inner_mlps=2):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands.
                     It represents the dim of input to MLP, augmented with positional encoding
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
            modulate: whether to apply exponential modulation
            w: frequency of periodic activations

        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, \
            "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = ComplexExponentialPositionalEmbedding(emb_dim, seq_len)

        # uses a variable number of inner linear layers
        self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, order), act)
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)
        # final linear layer
        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        self.modulate = modulate
        if self.modulate:
            self.modulation = ExponentialModulation(d_model)  # TODO add other args

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, padding_mask=None, causal=True):
        if k is None:
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias
        if padding_mask is None or causal:
            y = fftconv_ref(x, k, bias, gelu=False, dropout=self.dropout, causal=causal)
        else:
            seqlen = padding_mask.shape[1]
            b_idx = 0
            new_y = []
            for b_padding in torch.sum(padding_mask, dim=1):
                y_b = fftconv_ref(
                    x[b_idx:b_idx + 1, ..., :(seqlen - b_padding)],
                    k[..., :(seqlen - b_padding)],
                    bias,
                    dropout=self.dropout,
                    gelu=False,
                    causal=causal)
                new_y.append(F.pad(y_b, (0, b_padding)))
                b_idx += 1
            y = torch.cat(new_y)
        return y


class HyenaOperator(nn.Module):
    """
    Hyena operator described in the paper:
    `Hyena Hierarchy: Towards Larger Convolutional Language Models <https://arxiv.org/pdf/2302.10866.pdf>`_

    In the original implementation, the  positional embeddings in the Hyena filter use a custom (lower)
    learning rate different from the one used for the rest of the architecture (i.e. 1e-5 instead of 1e-3).
    We do not add this in our implementation, but we may want to test it in the future.
    """
    def __init__(
            self,
            d_model,
            l_max,
            order=2,
            filter_order=64,
            num_heads=1,
            inner_factor=1,
            num_blocks=1,
            outer_mixing=False,
            dropout=0.0,
            filter_dropout=0.0,
            post_order_ffn=False,
            short_filter_order=3,
            causal=True,
    ):
        r"""
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
        """
        super().__init__()
        assert d_model % num_heads == 0, f'Model dimension {d_model} must be divisible by num heads {num_heads}'
        assert l_max % num_blocks == 0, f'Maximum signal length {l_max} must be divisible by block dimension {num_blocks}'
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads
        self.d_model = d_model
        self.order = order
        self.l_max = l_max
        self.num_heads = num_heads
        self.inner_factor = inner_factor
        self.block_dim = block_dim
        self.head_dim = head_dim
        self.filter_order = filter_order
        self.post_order_ffn = post_order_ffn
        self.short_filter_order = short_filter_order
        self.num_blocks = num_blocks
        self.filter_dropout = filter_dropout
        self.outer_mixing = outer_mixing
        self.dropout = nn.Dropout(dropout)
        # setup projections: Initializes input and output projections (over the width dimension)
        self.out_proj = nn.Linear(self.d_model * inner_factor, self.d_model)
        self.in_proj = nn.Linear(self.d_model, (self.order + 1) * self.d_model)
        if self.post_order_ffn:
            self.ord_proj_w = nn.Parameter(
                torch.randn(self.order, self.num_heads, self.num_heads) / math.sqrt(self.head_dim))

        # setup filters: Initializes the explicit and implicit filters
        assert self.order >= 2, f'Order must be at least 2, (got {self.order})'
        total_width = self.d_model * self.inner_factor * (self.order + 1)
        self.causal = causal
        if self.causal:
            self.short_filter_padding = self.short_filter_order - 1
        else:
            self.short_filter_padding = self.short_filter_order // 2
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=self.short_filter_order,
            groups=total_width,
            padding=self.short_filter_padding)
        # TODO: support other configs
        self.filter_fn = HyenaFilter(
            self.head_dim * self.inner_factor * (self.order - 1),
            order=self.filter_order,
            seq_len=self.l_max,
            dropout=self.filter_dropout)

    def forward(self, u, padding_mask):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u.masked_fill_(padding_mask.unsqueeze(-1), 0)
        u = u.transpose(1, 2)  # b l d -> b d l
        uc = self.short_filter(u)[..., :l_filter]
        # b (ho v) (z l) -> b ho v z l
        uc = uc.reshape(uc.shape[0], self.num_heads, self.head_dim * (self.order + 1), self.num_blocks, -1)
        uc.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), 0)

        *x, v = uc.split(self.head_dim, dim=2)
        k = self.filter_fn.filter(l_filter)

        # `c` is always 1 by default
        # c l (v o) -> c o v l
        c, l, _ = k.shape
        k = k.transpose(1, 2).view(c, self.order - 1, self.head_dim, l)[0]

        bias = self.filter_fn.bias.view(self.head_dim, self.order - 1).transpose(0, 1)  # (v o) -> o v

        for o, x_i in enumerate(reversed(x[1:])):
            if self.outer_mixing:
                v = v.unsqueeze(2)  # b h v z l -> b h 1 v z l
                v = self.dropout(v * x_i.unsqueeze(2))  # b h v z l -> b h v 1 z l
                v = v.sum(dim=2)
            else:
                v = self.dropout(v * x_i)

            # the bias term is broadcasted. Last dimension (l) is handled by fftconv
            v = self.filter_fn(
                v, l_filter, k=k[o], bias=bias[o, None, :, None], padding_mask=padding_mask, causal=self.causal)
            v.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), 0)
            if self.post_order_ffn:
                w = self.ord_proj_w[o]
                w = w.unsqueeze(2).unsqueeze(2).unsqueeze(2).unsqueeze(0)  # h1 h2 -> 1 h1 h2 1 1 1
                v = v.unsqueeze(2)  # b h v z l -> b h 1 v z l
                v = mul_sum(w, v)

        y = v * x[0]
        # b h v z l -> b (z l) (h v)
        b, _, v, _, l = y.shape
        y = y.permute(0, 3, 4, 1, 2).view(b, self.num_blocks * l, self.num_heads * v)
        y = self.out_proj(y)
        y.masked_fill_(padding_mask.unsqueeze(-1), 0)
        return y
