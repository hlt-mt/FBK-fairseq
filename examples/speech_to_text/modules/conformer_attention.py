# This code was inspired to the Soohwan Kim (https://github.com/sooftware/conformer) repository.
# Read carefully their licence.

# Copyright 2021 FBK

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
from torch import Tensor
from typing import Optional

from torch.nn import init

from fairseq.modules import FairseqDropout


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    The version implemented on Fairseq differs slightly from the paper, this implementation is faithful to the
    original one. Please see
    :func:`~fairseq.modules.sinusoidal_positional_embedding.SinusoidalPositionalEmbedding.get_embedding` for more
    details.
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the `"Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    <https://arxiv.org/pdf/1901.02860.pdf>`_.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        query (batch, time, dim): Tensor containing query vector
        key (batch, time, dim): Tensor containing key vector
        value (batch, time, dim): Tensor containing value vector
        pos_embedding (batch, time, dim): Positional embedding tensor
        mask (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
            batch_unsafe_relative_shift: bool = False
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        init.xavier_uniform_(self.query_proj.weight)
        init.zeros_(self.query_proj.bias)
        self.key_proj = nn.Linear(d_model, d_model)
        init.xavier_uniform_(self.key_proj.weight)
        init.zeros_(self.key_proj.bias)
        self.value_proj = nn.Linear(d_model, d_model)
        init.xavier_uniform_(self.value_proj.weight)
        init.zeros_(self.value_proj.bias)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        init.xavier_uniform_(self.pos_proj.weight)

        self.dropout = FairseqDropout(p=dropout_p, module_name=self.__class__.__name__)
        # u and v are the trainable parameters of the Transformer-XL attention computation
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)
        init.xavier_uniform_(self.out_proj.weight)
        init.zeros_(self.out_proj.bias)
        self.relative_shift_func = self._relative_shift_unsafe if batch_unsafe_relative_shift else self._relative_shift

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        # Attention weights computation using Q + u as in Transformer-XL
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        # Relative positional weights computation using Q + v as in Transformer-XL
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        # Right shifting mechanism described in Transformer-XL
        pos_score = self.relative_shift_func(pos_score, mask)
        # Final attention weights obtained summing the attention with its relative positional embeddings
        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9 if mask.dtype == torch.float32 else -1e4)

        attn = F.softmax(score, dim=-1)
        # set to 0.0 all attention weights of padding elements
        if mask is not None:
            attn = attn.masked_fill(mask, 0.0)
        attn = self.dropout(attn)

        # Attention computation
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor, padding_mask: Tensor) -> Tensor:
        """
        This methods performs the relative shift operation row-wise.
        Although inefficient, it enforces that each row is shifted accounting its padding,
        which enforces that the result does not change depending on whether a given row
        is padded or not.
        """
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        assert seq_length1 == seq_length2, "Currently we support only self-attention"
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        seq_lengths = (seq_length1 - (padding_mask[:, :, 0]).sum(-1)).tolist()
        for b_i in range(batch_size):
            padded_batch_pos_scores = padded_pos_score[b_i, :, :seq_lengths[b_i], :seq_lengths[b_i] + 1]
            padded_batch_pos_scores = padded_batch_pos_scores.reshape(num_heads, seq_lengths[b_i] + 1, seq_lengths[b_i])
            pos_score[b_i, :, :seq_lengths[b_i], :seq_lengths[b_i]] = padded_batch_pos_scores[:, 1:, :]
        pos_score.masked_fill_(padding_mask.unsqueeze(1), 0.0)
        return pos_score

    def _relative_shift_unsafe(self, pos_score: Tensor, padding_mask: Tensor) -> Tensor:
        """
         This implementation reflects other open source ones (e.g. fairseq), which
         shift the values from the row above in the batch. Although efficient,
         this leads to inconsistencies in the results, as the same row has different
         values according to whether it is padded (and how much it is) or not.
         """
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        x (batch, time, dim): Tensor containing input vector
        mask (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, batch_unsafe_relative_shift: bool = False):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p, batch_unsafe_relative_shift)
        self.dropout = FairseqDropout(p=dropout_p, module_name=self.__class__.__name__)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = x.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        x = self.layer_norm(x)
        outputs = self.attention(x, x, x, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)
