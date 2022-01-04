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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from examples.linformer.linformer_src.modules.conv1d_compress import Conv1dCompressLayer
from examples.speech_to_text.models.conformer import conformer_base_architecture, conformer_s
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel
from examples.speech_to_text.modules.conformer_encoder_layer import ConformerEncoderLayer
from examples.speech_to_text.modules.ctc_support import CtcSupport
from examples.speech_to_text.modules.encoder_pretraining_support import EncoderPretrainingSupport
from examples.speech_to_text.modules.speechformer_encoder_layer import SpeechformerEncoderLayer
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable, Conv1dSubsampler
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding, TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)


class InitialConv1dBlock(Conv1dSubsampler):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.stride = stride
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=self.stride,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / self.stride + 1).floor().long()
        return out


@register_model("speechformer")
class SpeechformerModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        parser.add_argument(
            "--compressed", type=int, default=4,
            help="compression factor"
        )
        parser.add_argument(
            "--shared-kv-compressed",
            default=True, action='store_true',
            help="share compressed matrix between k and v, in each layer",
        )
        parser.add_argument(
            "--shared-layer-kv-compressed",
            default=True, action='store_true',
            help="share compressed matrix between k and v and across all layers",
        )
        parser.add_argument(
            "--freeze-compress",
            default=False, action='store_true',
            help="freeze the parameters in compressed layer",
        )
        parser.add_argument(
            "--compress-kernel-size",
            type=int,
            help="kernel size of the convolution used to compress the sequence length in the ConvAttn. "
                 "Note: it should be higher than (--compressed) parameter",
        )
        parser.add_argument(
            "--compress-n-layers",
            type=int, default=1,
            help="Number of layers used for the K and V projection in the ConvAttention layer",
        )
        parser.add_argument(
            '--transformer-after-compression', default=False, action='store_true',
            help='whether or not using standard TransformerEncoder layers after CTC compression '
            'instead of ConvAttention Encoder layers')
        parser.add_argument(
            '--conformer-after-compression', default=False, action='store_true',
            help='whether or not using ConformerEncoder layers after CTC compression '
                 'instead of ConvAttention Encoder layers')
        # Initial convolutional layer (optional)
        parser.add_argument(
            "--CNN-first-layer",
            default=True, action="store_true",
            help="if enabled, substitutes the initial linear layer with a couple of 1D "
                 "Convolutional layers"
        )
        parser.add_argument(
            "--stride",
            type=int,
            default=1,
            help="stride value in the initial Conv1d block"
        )

    @classmethod
    def build_encoder(cls, args, dictionary):
        encoder = SpeechformerEncoder(args, dictionary)
        encoder = EncoderPretrainingSupport.load_pretrained(args, encoder)
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, src_dict if src_dict is not None else tgt_dict)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, return_all_hiddens=True)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if self.encoder.ctc_flag:
            return decoder_out, {"ctc_out": encoder_out["ctc_out"], "ctc_lengths": encoder_out["ctc_lengths"]}
        else:
            return decoder_out


class SpeechformerEncoder(FairseqEncoder, CtcSupport):
    """Speechformer encoder
    It consists of:
        - if --CNN-first-layer parameter is enabled, a block of 2 1D Convolutional layers is
        inserted, otherwise a Linear layer is used to adapt the input dimension to the
        Speechformer embedding dimension
        - if --transformer-after-compression is set the standard TransformerEncoder layers are
        inserted after the CTC compression layer, otherwise the PlainConvattention architecture
        is obtained

    To use the Speechformer architecture of the paper "Speechformer: Reducing Information
    Loss in Direct Speech Translation", both --CNN-first-layer and --transformer-after-compression
    have to be enabled. To use the PlainConvattention architecture, only --CNN-first-layer has
    to be set.
    """

    def __init__(self, args, dictionary):
        self.compress_layer = None
        self.CNN_first_layer = args.CNN_first_layer

        super().__init__(dictionary)

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        if self.CNN_first_layer:
            self.CNNblock = InitialConv1dBlock(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                args.stride,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )
        else:
            self.linear_layer = nn.Linear(args.input_feat_per_channel * args.input_channels, args.encoder_embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        assert args.compress_kernel_size >= args.compressed
        transformer_after_compression = getattr(args, "transformer_after_compression", False)
        conformer_after_compression = getattr(args, "conformer_after_compression", False)
        assert not (transformer_after_compression and conformer_after_compression), \
            "Cannot enable both transformer_after_compression and conformer_after_compression"
        if transformer_after_compression or conformer_after_compression:
            self.speechformer_layers = nn.ModuleList(
                [self.build_speechformer_encoder_layer(args) for _ in range(args.ctc_encoder_layer)]
            )
            layer_class = TransformerEncoderLayer if transformer_after_compression else ConformerEncoderLayer
            self.speechformer_layers.extend(
                [layer_class(args) for _ in range(args.encoder_layers - args.ctc_encoder_layer)]
            )
        else:
            self.speechformer_layers = nn.ModuleList(
                [self.build_speechformer_encoder_layer(args) for _ in range(args.encoder_layers)]
            )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.ctc_init(args, dictionary)

    def build_speechformer_encoder_layer(self, args):
        if args.shared_layer_kv_compressed == 1 and self.compress_layer is None:
            compress_layer = Conv1dCompressLayer(
                args.encoder_embed_dim,
                args.compress_kernel_size,
                compression_factor=args.compressed,
                padding=args.compress_kernel_size // 2,
                n_layers=args.compress_n_layers,
                freeze_compress=args.freeze_compress,
            )
            self.compress_layer = compress_layer

        return SpeechformerEncoderLayer(args, self.compress_layer)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k not in ["prev_output_tokens", "prev_transcript_tokens"]
        }
        return self.forward(**encoder_input)

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **kwargs):
        if self.CNN_first_layer:
            x, input_lengths = self.CNNblock(src_tokens, src_lengths)
        else:
            x = self.linear_layer(src_tokens)
            input_lengths = src_lengths
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if self.CNN_first_layer:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            x = self.dropout_module(x)
        else:
            positions = self.embed_positions(encoder_padding_mask)
            x += positions
            x = self.dropout_module(x).transpose(0, 1)

        encoder_states = []

        x_ctc = None
        ctc_lengths = input_lengths
        for l_idx, layer in enumerate(self.speechformer_layers):
            x = layer(x, encoder_padding_mask)
            # ctc
            if self.ctc_flag and self.ctc_layer == l_idx + 1:
                x, x_ctc, encoder_padding_mask = self.apply_ctc(x, input_lengths)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

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
        """Reorder encoder output according to *new_order*.

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


@register_model_architecture(model_name="speechformer", arch_name="speechformer")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.max_seq_len = getattr(args, "max_seq_len", 4096)
    args.add_position_after_ctc = getattr(args, "add_position_after_ctc", False)

    # Compression parameters
    args.compressed = getattr(args, "compressed", 4)
    args.shared_kv_compressed = getattr(args, "shared_kv_compressed", True)
    args.shared_layer_kv_compressed = getattr(args, "shared_layer_kv_compressed", True)
    args.compress_n_layers = getattr(args, "compress_n_layers", 1)
    args.freeze_compress = getattr(args, "freeze_compress", False)
    args.compress_kernel_size = getattr(args, "compress_kernel_size", 8)
    args.CNN_first_layer = getattr(args, "CNN_first_layer", True)

    # Optional convolution layers parameters
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    args.stride = getattr(args, "stride", 1)


@register_model_architecture("speechformer", "speechformer_s")
def speechformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("speechformer", "speechformer_m")
def speechformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("speechformer", "speechformer_l")
def speechformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("speechformer", "speechconformer_s")
def speechconformer_s(args):
    speechformer_s(args)
    conformer_s(args)


@register_model_architecture("speechformer", "speechconformer_m")
def speechconformer_m(args):
    speechformer_m(args)
    conformer_base_architecture(args)
