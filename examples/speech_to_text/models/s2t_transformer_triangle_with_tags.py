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

from examples.speech_to_text.data.speech_to_text_dataset_tagged import S2TDataConfigTagged
from examples.speech_to_text.models.base_triangle import TriangleTransformerDecoder
from examples.speech_to_text.models.s2t_transformer_fbk_triangle import S2TTransformerTriangle
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, base_architecture, \
    s2t_transformer_m, s2t_transformer_s
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text import TransformerDecoderScriptable


@register_model('s2t_transformer_triangle_with_tags')
class S2TTransformerTriangleWithTags(S2TTransformerTriangle):
    """
    This model is an implementation of a multi-task model that predicts both transcripts
    and translations, with the translation being generated from the output representation
    of the transcript decoder. It represents the triangle model of (Sperber et al. 2020).
    In addition, it outputs for each token in both transcripts and translations
    the corresponding tag information (e.g. the named entity type).
    """
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 100000
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 100000

        # This model requires a task that provides source dictionary and transcripts
        assert task.source_dictionary is not None and task.target_dictionary is not None
        # and a data config containing the tags
        assert isinstance(task.data_cfg, S2TDataConfigTagged)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        target_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        src_embed_tokens = build_embedding(src_dict, args.decoder_embed_dim)
        encoder = S2TTransformerModel.build_encoder(args, src_dict)
        tags_num = len(task.data_cfg.tags) + 1  # all tags plus the no-tag case
        decoder = TriangleTransformerDecoderWithTags(args, tgt_dict, target_embed_tokens, tags_num)
        auxiliary_decoder = TransformerDecoderWithTags(args, src_dict, src_embed_tokens, tags_num)
        return S2TTransformerTriangleWithTags(encoder, decoder, auxiliary_decoder)


class TransformerDecoderWithTags(TransformerDecoderScriptable):
    def __init__(self, args, dictionary, embed_tokens, tags_num):
        super().__init__(args, dictionary, embed_tokens)
        self.tags_output_projection = nn.Linear(self.output_embed_dim, tags_num, bias=False)
        nn.init.normal_(self.tags_output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
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
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        extra = {"tags": self.tags_output_projection(x), "attn": None}
        if not features_only:
            x = self.output_layer(x)
        return x, extra


class TriangleTransformerDecoderWithTags(TriangleTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, tags_num):
        super().__init__(args, dictionary, embed_tokens)
        self.tags_output_projection = nn.Linear(self.output_embed_dim, tags_num, bias=False)
        nn.init.normal_(self.tags_output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5)

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
        extra["tags"] = self.tags_output_projection(x)
        if not features_only:
            x = self.output_layer(x)
        return x, extra


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture('s2t_transformer_triangle_with_tags', 's2t_transformer_triangle_with_tags')
def base_triangle_with_tags_architecture(args):
    base_architecture(args)


@register_model_architecture('s2t_transformer_triangle_with_tags', 's2t_transformer_triangle_with_tags_s')
def s2t_triangle_with_tags_s(args):
    s2t_transformer_s(args)


@register_model_architecture('s2t_transformer_triangle_with_tags', 's2t_transformer_triangle_with_tags_m')
def s2t_triangle_with_tags_m(args):
    s2t_transformer_m(args)
