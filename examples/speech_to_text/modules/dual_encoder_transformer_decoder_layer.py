# Copyright 2022 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Optional, Dict, List

import torch
from torch import Tensor
from torch.nn import Linear

from fairseq.modules import MultiheadAttention, TransformerDecoderLayer, LayerNorm


class TransformerDualEncoderDecoderLayer(TransformerDecoderLayer):
    """
    Decoder layer block that processes two encoder outputs instead of a single one.

    As per "Contextualized Translation of Automatically Segmented Speech" by Gaido et al. 2020
    (see https://www.isca-speech.org/archive/pdfs/interspeech_2020/gaido20_interspeech.pdf),
    the cross-attention can be computed in "parallel", where the two cross-attentions of the
    two encoders are computed individually starting from `encoder_out` and `context_encoder_out`
    and then summed together, or in a "sequential" way, where the output of the first
    cross-attention obtained from the `encoder_out` is given as input to the second
    cross-attention that processes the `context_encoder_out`.
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            args, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        self.context_attention_type = args.context_decoder_attention_type
        self.context_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        self.context_gating_wi = Linear(self.embed_dim, self.embed_dim)
        self.context_gating_ws = Linear(self.embed_dim, self.embed_dim)
        if self.context_attention_type == "sequential":
            self.context_attn_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        context_encoder_out: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            context_encoder_out (Tensor): external context to the layer of shape `(context_len, batch, embed_dim)`
            context_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, context_len)` where padding elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            # If parallel context attention is enabled, we need the normalized
            # input for the context cross-attention
            if self.context_attention_type == "parallel":
                query_context = x
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        # Context attention
        if self.context_attention_type == "sequential":
            residual = x
            if self.normalize_before:
                x = self.context_attn_layer_norm(x)
            x, _ = self.context_attn(
                query=x, key=context_encoder_out, value=context_encoder_out, key_padding_mask=context_padding_mask,
                incremental_state=incremental_state, static_kv=True)
            x = self.dropout_module(x)
            lambda_gating = torch.sigmoid(self.context_gating_wi(residual) + self.context_gating_ws(x))
            x = lambda_gating * residual + (1 - lambda_gating) * x
            if not self.normalize_before:
                x = self.context_attn_layer_norm(x)
        elif self.context_attention_type == "parallel":
            context_x, _ = self.context_attn(
                query=query_context, key=context_encoder_out, value=context_encoder_out,
                key_padding_mask=context_padding_mask, incremental_state=incremental_state, static_kv=True)
            context_x = self.dropout_module(context_x)
            lambda_gating = torch.sigmoid(self.context_gating_wi(x) + self.context_gating_ws(context_x))
            x = lambda_gating * x + (1 - lambda_gating) * context_x
        else:
            raise RuntimeError(
                "Invalid decoder context attention type {}".format(self.context_attention_type))

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None
