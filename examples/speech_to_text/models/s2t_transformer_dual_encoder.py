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

import logging
import copy
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from examples.speech_to_text.models.dual_encoder import FairseqDualEncoderModel
from fairseq.models.transformer import base_architecture as transformer_base_architecture
from examples.speech_to_text.models.s2t_transformer_fbk import base_architecture as s2t_transformer_fbk_base_architecture
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, S2TTransformerEncoder, \
    s2t_transformer_s, s2t_transformer_m, s2t_transformer_l
from examples.speech_to_text.modules.dual_encoder_transformer_decoder_layer import TransformerDualEncoderDecoderLayer
from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerEncoder
from fairseq.modules import FairseqDropout

logger = logging.getLogger(__name__)


@register_model("s2t_transformer_dual_encoder")
class S2TTransformerDualEncoderModel(FairseqDualEncoderModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        parser.add_argument('--context-encoder-layers', type=int, metavar='N',
                            help='num context encoder layers')
        parser.add_argument('--context-dropout', type=float, metavar='N', default=0.3,
                            help='context encoder dropout')
        parser.add_argument('--context-ffn-embed-dim', type=int, metavar='N', default=1024,
                            help='context encoder FFN embed dim')
        parser.add_argument('--context-decoder-attention-type', type=str, default="sequential",
                            choices=['parallel', "sequential"])
        parser.add_argument('--pretrained-model', type=str, default=None,
                            help='path to a pretrained ST model')
        parser.add_argument('--share-encoder-decoder-embed', action='store_true', default=False,
                            help='share encoder and decoder embeddings,'
                                 'source dictionary is used to build the embeddings')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 100000
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 100000

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        encoder_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim)
        if args.share_encoder_decoder_embed:
            decoder_embed_tokens = encoder_embed_tokens
        else:
            decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim)

        context_args = copy.deepcopy(args)
        context_args.encoder_ffn_embed_dim = args.context_ffn_embed_dim
        context_args.encoder_layers = args.context_encoder_layers
        context_args.dropout = args.context_dropout
        context_encoder = TransformerEncoder(context_args, src_dict, encoder_embed_tokens)
        encoder = S2TTransformerEncoder(args, src_dict)
        decoder = TransformerDualEncoderDecoder(args, tgt_dict, decoder_embed_tokens)
        model = S2TTransformerDualEncoderModel(encoder, decoder, context_encoder)

        if args.pretrained_model is not None:
            pretrained_model_state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_model)
            incompatible_keys = model.load_state_dict(pretrained_model_state["model"], strict=False)
            if len(incompatible_keys.unexpected_keys) != 0:
                logger.error("Cannot load the following keys from checkpoint: {}".format(
                    incompatible_keys.unexpected_keys))
                raise ValueError("Cannot load from checkpoint: {}".format(args.pretrained_model))

            if getattr(args, 'freeze_pretrained', False):
                for p_name, p_val in model.named_parameters():
                    if p_name in pretrained_model_state["model"]:
                        p_val.requires_grad = False

        return model


class TransformerDualEncoderDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDualEncoderDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.n_decoder_layers = args.decoder_layers
        args.decoder_layers = 0  # To avoid generating useless layers overridden later
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        args.decoder_layers = self.n_decoder_layers
        self.layers = nn.ModuleList([
            TransformerDualEncoderDecoderLayer(args) for _ in range(self.n_decoder_layers)
        ])
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        context_encoder_out: Optional[Dict[str, torch.Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            context_encoder_out (optional): output from the context encoder, used
                for context attention
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
            context_encoder_out=context_encoder_out,
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
        encoder_out: Optional[Dict[str, List[torch.Tensor]]],
        context_encoder_out: Optional[Dict[str, torch.Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
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

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[torch.Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[torch.Tensor] = None
        inner_states: List[Optional[torch.Tensor]] = [x]
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
                incremental_state,
                context_encoder_out=context_encoder_out['encoder_out'][0]
                if context_encoder_out is not None else None,
                context_padding_mask=context_encoder_out['encoder_padding_mask'][0]
                if context_encoder_out is not None else None,
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


@register_model_architecture('s2t_transformer_dual_encoder', 's2t_transformer_dual_encoder')
def base_architecture(args):
    s2t_transformer_fbk_base_architecture(args)
    transformer_base_architecture(args)
    args.context_encoder_layers = getattr(args, "context_encoder_layers", args.encoder_layers)
    args.context_dropout = getattr(args, "context_dropout", 0.3)
    args.context_ffn_embed_dim = getattr(args, "context_ffn_embed_dim", 1024)
    args.context_decoder_attention_type = getattr(args, "context_decoder_attention_type", "parallel")
    args.share_encoder_decoder_embed = getattr(args, "share_encoder_decoder_embed", False)


@register_model_architecture('s2t_transformer_dual_encoder', 's2t_transformer_dual_encoder_s')
def s2t_transformer_dual_encoder_s(args):
    s2t_transformer_s(args)


@register_model_architecture('s2t_transformer_dual_encoder', 's2t_transformer_dual_encoder_m')
def s2t_transformer_dual_encoder_m(args):
    s2t_transformer_m(args)


@register_model_architecture('s2t_transformer_dual_encoder', 's2t_transformer_dual_encoder_l')
def s2t_transformer_dual_encoder_l(args):
    s2t_transformer_l(args)
