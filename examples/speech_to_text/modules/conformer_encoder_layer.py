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

import torch.nn as nn
from torch import Tensor

from examples.speech_to_text.modules.conformer_attention import MultiHeadedSelfAttentionModule
from examples.speech_to_text.modules.conformer_convolution import (
    ConformerConvModule,
)
from examples.speech_to_text.modules.conformer_feedforward import FeedForwardModule


class ConformerEncoderLayer(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        x (time, batch, dim): Tensor containing input vector

    Returns: outputs
        **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(self, args):
        super().__init__()
        self.encoder_dim = args.encoder_embed_dim
        self.num_attention_heads = args.encoder_attention_heads
        self.feed_forward_expansion_factor = args.feed_forward_expansion_factor
        self.conv_expansion_factor = args.conv_expansion_factor
        self.feed_forward_dropout_p = args.conformer_feedforward_dropout
        self.attention_dropout_p = args.conformer_attention_dropout
        self.conv_dropout_p = args.conformer_conv_dropout
        self.conv_kernel_size = args.conformer_conv_kernel_size
        self.half_step_residual = args.conformer_half_step_residual
        self.no_syncbatchnorm = args.no_syncbatchnorm
        self.batch_unsafe_relative_shift = getattr(args, 'batch_unsafe_relative_shift', False)

        if self.half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.first_feed_forward = FeedForwardModule(
            encoder_dim=self.encoder_dim,
            expansion_factor=self.feed_forward_expansion_factor,
            dropout_p=self.feed_forward_dropout_p,
        )

        self.attention = MultiHeadedSelfAttentionModule(
            d_model=self.encoder_dim,
            num_heads=self.num_attention_heads,
            dropout_p=self.attention_dropout_p,
            batch_unsafe_relative_shift=self.batch_unsafe_relative_shift,
        )

        self.conv_module = ConformerConvModule(
            in_channels=self.encoder_dim,
            kernel_size=self.conv_kernel_size,
            expansion_factor=self.conv_expansion_factor,
            dropout_p=self.conv_dropout_p,
            no_syncbatchnorm=self.no_syncbatchnorm,
        )

        self.second_feed_forward = FeedForwardModule(
            encoder_dim=self.encoder_dim,
            expansion_factor=self.feed_forward_expansion_factor,
            dropout_p=self.feed_forward_dropout_p,
        )

        self.layernorm = nn.LayerNorm(self.encoder_dim)

    def forward(self, x: Tensor, encoder_padding_mask: Tensor) -> Tensor:
        x = x.transpose(0, 1)  # B x T x C
        new_x = self.first_feed_forward(x)
        x = new_x * self.feed_forward_residual_factor + x
        # we need attention padding mask (attn_mask) to be applied during the attention calculation,
        # we obtain it from the encoder_padding_mask (B x T) by repeating it T times (x.shape[1]) and
        # taking the logical or to correctly mask both T x T dimensions
        att_mask = encoder_padding_mask.unsqueeze(1).repeat([1, x.shape[1], 1])
        att_mask = att_mask.logical_or(att_mask.transpose(1, 2))    # B x T x T
        new_x = self.attention(x, att_mask)
        x = new_x + x
        new_x = self.conv_module(x, encoder_padding_mask)
        x = new_x + x
        new_x = self.second_feed_forward(x)
        x = new_x * self.feed_forward_residual_factor + x
        return self.layernorm(x).transpose(1, 0)
