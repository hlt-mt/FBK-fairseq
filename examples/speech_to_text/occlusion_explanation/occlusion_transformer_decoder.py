# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Optional, List, Tuple, Dict

from torch import Tensor, LongTensor

from examples.speech_to_text.occlusion_explanation.decoder_perturbator import \
    OcclusionDecoderEmbeddingsPerturbator
from fairseq.data import Dictionary
from fairseq.models.speech_to_text import TransformerDecoderScriptable
from fairseq.models.transformer import Embedding


class OcclusionTransformerDecoderScriptable(TransformerDecoderScriptable):
    """
    Transformer decoder class where occlusion of target token embeddings is performed.
    Occlusion can be performed either before or after the sum with position embeddings,
    and can be performed either in a continuous way, namely for single values independently,
    or in a discrete way, namely by zeroing out entire embeddings of single tokens.
    The number of masks is determined by the number of masks in fbank_occlusion.
    """
    def __init__(
            self,
            args,
            dictionary: Dictionary,
            perturbator: OcclusionDecoderEmbeddingsPerturbator,
            no_encoder_attn: bool = False):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        embed_tokens = Embedding(num_embeddings, args.decoder_embed_dim, padding_idx)
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.perturbator = perturbator

    def embed_tokens_positions(
            self, prev_output_tokens: LongTensor, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns decoder input embeddings with positions applying occlusion.
        """
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.perturbator.no_position_occlusion:
            x, masks = self.perturbator(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        if not self.perturbator.no_position_occlusion:
            x, masks = self.perturbator(x)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        return x, masks

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None) -> Tuple[Tensor, Dict]:
        """
        Overrides extract_features_scriptable() of TransformerDecoder
        to implement occlusion of decoder input emebddings.
        Args:
            - full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            - alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            - alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state)
            if self.embed_positions is not None
            else None)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        x, masks = self.embed_tokens_positions(prev_output_tokens, positions)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0)
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)))
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

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"masks": masks, "attn": [attn], "inner_states": inner_states}

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None):
        x, model_specific_output = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads)
        if alignment_layer is None:
            return x, model_specific_output
        return x, model_specific_output
