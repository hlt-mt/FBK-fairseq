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
from typing import Optional, Dict, Any, List

from torch import nn, Tensor
import torch

from examples.speech_to_text.models.multi_task import MultiTaskModel
from examples.speech_to_text.modules.triangle_transformer_layer import TriangleTransformerDecoderLayer
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text import TransformerDecoderScriptable, FairseqEncoderDecoderModel


class BaseTriangle(MultiTaskModel):
    """
    This model is an implementation of a multi-task model that predicts both transcripts
    and translations, with the translation being generated from the output representation
    of the transcript decoder. It represents the triangle model of (Sperber et al. 2020).
    """
    # TODO: do we need different settings/configs for the two decoders?
    # For now, we assume NO.

    encoder_parent_model: FairseqEncoderDecoderModel

    @staticmethod
    def add_base_args(args):
        pass

    @staticmethod
    def add_args(parser):
        parser.add_argument('--auxiliary-decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        MultiTaskModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        cls.add_base_args(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 100000
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 100000

        # This model requires a task that provides source dictionary and transcripts
        assert task.source_dictionary is not None and task.target_dictionary is not None

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        target_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        src_embed_tokens = build_embedding(src_dict, args.decoder_embed_dim)
        encoder = cls.encoder_parent_model.build_encoder(args, src_dict if src_dict is not None else tgt_dict)
        decoder = TriangleTransformerDecoder(args, tgt_dict, target_embed_tokens)
        auxiliary_decoder = TransformerDecoderScriptable(args, src_dict, src_embed_tokens)
        return cls(encoder, decoder, auxiliary_decoder, args)

    # In "speech_translation_with_transcription" the transcripts are read into
    # "transcript_target". Not the most elegant solution, but it allows
    # compatibility with existing code.
    def get_auxiliary_target(self, sample, auxiliary_output):
        return sample["transcript"]

    def get_auxiliary_token_lens(self, sample):
        return sample["transcript_lengths"]

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_transcript_tokens, **kwargs):
        for k in ['prev_target_tags', 'prev_transcript_tags']:
            if k in kwargs:
                del kwargs[k]

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        auxiliary_out = self.auxiliary_decoder(
            prev_transcript_tokens, encoder_out=encoder_out, features_only=True)
        auxiliary_padding_mask = prev_transcript_tokens.eq(
            self.auxiliary_decoder.padding_idx)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            aux_decoder_out=auxiliary_out[0].transpose(0, 1),
            aux_decoder_padding_mask=auxiliary_padding_mask,
            **kwargs
        )
        if self.encoder.ctc_flag:
            return (decoder_out, (self.auxiliary_decoder.output_layer(auxiliary_out[0]), auxiliary_out[1])), \
                   {"ctc_out": encoder_out["ctc_out"], "ctc_lengths": encoder_out["ctc_lengths"]}
        else:
            return decoder_out, (self.auxiliary_decoder.output_layer(auxiliary_out[0]), auxiliary_out[1])

    def forward_decoder(self, prev_output_tokens, encoder_out, auxiliary_out, auxiliary_tokens, **kwargs):
        auxiliary_padding_mask = auxiliary_tokens.eq(
            self.auxiliary_decoder.padding_idx)
        return self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            aux_decoder_out=auxiliary_out.transpose(0, 1),
            aux_decoder_padding_mask=auxiliary_padding_mask,
            **kwargs
        )


class TriangleTransformerDecoder(TransformerDecoderScriptable):
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TriangleTransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        aux_decoder_out: Optional[torch.Tensor] = None,
        aux_decoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            aux_decoder_out=aux_decoder_out,
            aux_decoder_padding_mask=aux_decoder_padding_mask,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, Optional[Tensor]]] = None,
        aux_decoder_out: Optional[torch.Tensor] = None,
        aux_decoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
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
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                aux_decoder_out,
                aux_decoder_padding_mask,
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

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
