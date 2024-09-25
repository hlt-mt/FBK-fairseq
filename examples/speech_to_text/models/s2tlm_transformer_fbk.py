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
import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor

from fairseq.models import register_model, register_model_architecture
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, \
    base_architecture, s2t_transformer_s
from fairseq.models.transformer import TransformerDecoder


logger = logging.getLogger(__name__)


@register_model("s2tlm_transformer_fbk")
class S2TLMTransformerModel(S2TTransformerModel):
    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        parser.add_argument(
            "--causal-prompt-mask", action="store_true", default=False,
            help="Apply causual masking on audio prompt"
        )

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return AudioPrependedTransformerDecoder(args, task.target_dictionary, embed_tokens, no_encoder_attn=True)


class AudioPrependedTransformerDecoder(TransformerDecoder):
    def __init__(self, args, task, embed_tokens, no_encoder_attn=False):
        super().__init__(args, task, embed_tokens, no_encoder_attn)
        self.prompt_causal_masking = getattr(args, "causal_prompt_mask", False)

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to the `TransformerDecoder` forward pass but concatenates the encoder output
        to the `prev_output_tokens` embeddings, so that the two information interact in the
        self-attention instead of with the cross-attention mechanism (that is disabled here).
        """
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        if (incremental_state is None or len(incremental_state) == 0) and not full_context_alignment:
            self_attn_mask = self.buffered_future_mask(x)
        else:
            self_attn_mask = None

        # Prepending encoder output (audio) to the embeddings of the text tokens.
        # This is done only at training time or in the first step of the generation
        # at inference time, as the prefix is cached for subsequent generations
        prepended_audio = False
        encoder_out_seq = encoder_out["encoder_out"][0]
        audio_len, batch_size, n_channels_audio = encoder_out_seq.shape
        if incremental_state is None or len(incremental_state) == 0:
            prepended_audio = True
            if len(encoder_out["encoder_padding_mask"]) > 0:
                encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
            else:
                encoder_padding_mask = self_attn_padding_mask.new_zeros(
                    batch_size, audio_len)
            x = torch.cat((encoder_out_seq, x), dim=0)  # T x B x C
            # update the padding mask to consider the prepended audio
            self_attn_padding_mask = torch.cat(
                (encoder_padding_mask, self_attn_padding_mask), dim=1)
            if self_attn_mask is not None:
                self_attn_mask = self.buffered_future_mask(x)
                if not self.prompt_causal_masking:
                    # as the future mask is cached, we need to avoid to perform the inplace
                    # operation (setting to 0) on the cached version, otherwise, in case the next
                    # batch has a shorter audio, the textual tokens will have access to the future
                    self_attn_mask = self_attn_mask.clone()
                    self_attn_mask[:audio_len, :audio_len] = 0.

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                None,
                None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if prepended_audio:
            # strip the prepended audio prefix, so we consider only the textual part for coherence
            # with the output of the cross-attention decoder
            x = x[:, encoder_out_seq.shape[0]:, :]
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn] if attn is not None else None, "inner_states": inner_states}


@register_model_architecture(model_name="s2tlm_transformer_fbk", arch_name="s2tlm_transformer_fbk")
def base_s2tlm_transformer_fbk(args):
    base_architecture(args)


@register_model_architecture("s2tlm_transformer_fbk", "s2tlm_transformer_s_fbk")
def s2tlm_transformer_s(args):
    s2t_transformer_s(args)
