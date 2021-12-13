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

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel
from examples.speech_to_text.models.s2t_transformer_fbk import base_architecture
from examples.speech_to_text.modules.conformer_encoder_layer import ConformerEncoderLayer
from examples.speech_to_text.modules.ctc_support import CtcSupport
from examples.speech_to_text.modules.encoder_pretraining_support import EncoderPretrainingSupport
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture, FairseqEncoder,
)
from fairseq.models.speech_to_text import Conv1dSubsampler, Dict
from fairseq.modules import FairseqDropout

logger = logging.getLogger(__name__)


@register_model("conformer")
class ConformerModel(S2TTransformerModel):
    """
    Conformer model from `"Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati, et al, 2020)
    <https://arxiv.org/abs/2005.08100>`_.
    This model implements the Conformer Encoder layer of the paper in the encoder while adopting the Transformer Decoder
    layer in the decoder. The 2D Convolutional layers before the Conformer Encoder are replaced by 1D Convolutional
    layers.
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Adds model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        parser.add_argument(
            '--feed-forward-expansion-factor', type=int, default=4,
            help='Expansion factor of Conformer FeedForward module'
        )
        parser.add_argument(
            '--conv-expansion-factor', type=int, default=2,
            help='Expansion factor of Conformer Convolution module'
        )
        parser.add_argument(
            '--conformer-feedforward-dropout', type=float, default=0.1,
            help='Dropout probability of the Conformer FeedForward module'
        )
        parser.add_argument(
            '--conformer-attention-dropout', type=float, default=0.1,
            help='Dropout probability of the Conformer Attention'
        )
        parser.add_argument(
            '--conformer-conv-dropout', type=float, default=0.1,
            help='Dropout probability of the Conformer Convolutional module'
        )
        parser.add_argument(
            '--conformer-conv-kernel-size', type=int, default=31,
            help='Kernel size of the Conformer Convolutional Layers'
        )
        parser.add_argument(
            '--conformer-half-step-residual', default=True, action='store_true',
            help='Whether to use half step residual or not in Conformer'
        )

    @classmethod
    def build_encoder(cls, args, dictionary):
        encoder = ConformerEncoder(args, dictionary)
        encoder = EncoderPretrainingSupport.load_pretrained(args, encoder)
        return encoder


class ConformerEncoder(FairseqEncoder, CtcSupport):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.conformer_layers = nn.ModuleList(
            [ConformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

        self.ctc_init(args, dictionary)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k not in ["prev_output_tokens", "prev_transcript_tokens"]
        }
        return self.forward(**encoder_input)

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **kwargs):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        x = self.dropout_module(x)

        encoder_states = []

        x_ctc = None
        ctc_lengths = input_lengths
        for l_idx, layer in enumerate(self.conformer_layers):
            x = layer(x, encoder_padding_mask)
            # ctc
            if self.ctc_flag and self.ctc_layer == l_idx + 1:
                x, x_ctc, encoder_padding_mask = self.apply_ctc(x, input_lengths)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        encoder_out_dict = {
            "encoder_out": [x],
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],
            "encoder_embedding": [],
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }
        return self.ctc_encoder_out(encoder_out_dict, x_ctc, ctc_lengths)

    def reorder_encoder_out(self, encoder_out, new_order):
        """
            Reorder encoder output according to *new_order*.

            Args:
                encoder_out: output from the ``forward()`` method
                new_order (LongTensor): desired order

            Returns:
                **encoder_out**: rearranged according to *new_order*

            The other things reordered a.t.m. are not mandatory
        """
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        reordered_dict = {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }
        return self.reorder_ctc(reordered_dict, encoder_out, new_order)


@register_model_architecture(model_name="conformer", arch_name="conformer")
def conformer_base_architecture(args):
    base_architecture(args)

    # Conformer parameters
    args.feed_forward_expansion_factor = getattr(args, "feed_forward_expansion_factor", 4)
    args.conv_expansion_factor = getattr(args, "conv_expansion_factor", 2)
    args.conformer_feedforward_dropout = getattr(args, "conformer_feedforward_dropout", 0.1)
    args.conformer_attention_dropout = getattr(args, "conformer_attention_dropout", 0.1)
    args.conformer_conv_dropout = getattr(args, "conformer_conv_dropout", 0.1)
    args.conformer_conv_kernel_size = getattr(args, "conformer_conv_kernel_size", 31)
    args.conformer_half_step_residual = getattr(args, "conformer_half_step_residual", True)


@register_model_architecture("conformer", "conformer_s")
def conformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    conformer_base_architecture(args)
