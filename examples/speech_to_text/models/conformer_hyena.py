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
import math
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from examples.speech_to_text.models.conformer import ConformerModel, conformer_base_architecture
from examples.speech_to_text.models.speechformer import InitialConv1dBlock
from examples.speech_to_text.modules.conformer_encoder_layer import ConformerEncoderLayer
from examples.speech_to_text.modules.conformer_hyena_encoder_layer import ConformerHyenaEncoderLayer
from examples.speech_to_text.modules.ctc_support import CtcSupport
from examples.speech_to_text.modules.encoder_pretraining_support import EncoderPretrainingSupport
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture, FairseqEncoder,
)
from fairseq.modules import FairseqDropout


logger = logging.getLogger(__name__)


@register_model("confhyena")
class ConformerHyenaModel(ConformerModel):
    """
    A model that has a Conformer encoder and Transformer decoder, in which the Conformer encoder
    uses the Hyena operator instead of the self-attention mechanism.

    The architecture is described in detail in:
    `How do Hyenas deal with Human Speech? Speech Recognition and Translation with ConfHyena <https://arxiv.org/abs/2402.13208>`_
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Adds model-specific arguments to the parser."""
        ConformerModel.add_args(parser)
        parser.add_argument(
            '--conformer-after-compression', default=False, action='store_true',
            help='whether or not using ConformerEncoder layers after CTC compression '
                 'instead of Hyena Encoder layers')
        parser.add_argument(
            '--hyena-order', type=int, default=2,
            help='order of Hyena operators'
        )
        parser.add_argument(
            '--hyena-filter-order', type=int, default=64,
            help='width of the FFN parametrizing the implicit filter in Hyena operators'
        )
        parser.add_argument(
            "--stride",
            type=int,
            default=1,
            help="stride value in the initial Conv1d block"
        )
        parser.add_argument(
            '--hyena-causal', default=False, action='store_true',
            help='if set, the Hyena operator is causal, as in the original implementation')

    @classmethod
    def build_encoder(cls, args, dictionary):
        encoder = ConformerHyenaEncoder(args, dictionary)
        encoder = EncoderPretrainingSupport.load_pretrained(args, encoder)
        return encoder


class ConformerHyenaEncoder(FairseqEncoder, CtcSupport):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = InitialConv1dBlock(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            stride=args.stride,
            kernel_sizes=[int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        if args.conformer_after_compression:
            self.conformer_layers = nn.ModuleList(
                [self.build_hyena_conformer_layer(args) for _ in range(args.ctc_encoder_layer)]
            )
            self.conformer_layers.extend(
                [ConformerEncoderLayer(args) for _ in range(args.encoder_layers - args.ctc_encoder_layer)]
            )
        else:
            self.conformer_layers = nn.ModuleList(
                [self.build_hyena_conformer_layer(args) for _ in range(args.encoder_layers)]
            )

        self.ctc_init(args, dictionary)

    def build_hyena_conformer_layer(self, args):
        return ConformerHyenaEncoderLayer(args)

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


@register_model_architecture(model_name="confhyena", arch_name="confhyena")
def conformer_hyena_base_architecture(args):
    conformer_base_architecture(args)
    args.conformer_after_compression = getattr(args, "conformer_after_compression", False)
    args.hyena_order = getattr(args, "hyena_order", 2)
    args.hyena_filter_order = getattr(args, "hyena_filter_order", 64)


@register_model_architecture("confhyena", "confhyena_s")
def conformer_hyena_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    conformer_hyena_base_architecture(args)
