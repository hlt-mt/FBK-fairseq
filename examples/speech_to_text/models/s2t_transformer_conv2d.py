#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple
from itertools import groupby

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils

from examples.speech_to_text.modules.conv_transformer_layer import ConvTransformerEncoderLayer
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable, S2TTransformerModel
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
)

import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Conv2dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        activation_fn,
        dropout,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
        stride: int = 2,
    ):
        super(Conv2dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(
                1 if i == 0 else mid_channels,
                mid_channels,
                k,
                stride,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )
        conv_out_channels = in_channels
        for _ in range(self.n_layers):
            conv_out_channels //= 2
        fc_input_features = mid_channels * conv_out_channels
        self.fc = nn.Linear(fc_input_features, out_channels)
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.bn = nn.ModuleList([BatchNorm(mid_channels) for _ in range(len(kernel_sizes))])

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        x = src_tokens.unsqueeze(1)
        for i, conv in enumerate(self.conv_layers):
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = self.bn[i](self.activation_fn(x))
            # src_lengths = torch.ceil(src_lengths.float() / 2).long()
            x = F.dropout(x, p=max(self.dropout, .1), training=self.training)
        # B x Cout x T x F -> T x B x C
        bsz, out_channels, time, feats = x.size()
        x = x.transpose(1, 2).contiguous().view(bsz, time, -1).contiguous().transpose(0, 1)
        x = self.activation_fn(self.fc(x))
        return x, self.get_out_seq_lens_tensor(src_lengths)


@register_model("s2t_transformer_conv2d")
class S2TTransformerModelConv2d(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        """Add model-specific arguments to the parser."""
        parser.add_argument('--distance-penalty', type=str, default=False,
                            choices=['log', 'gauss'],
                            help='Add distance penalty to the encoder')
        parser.add_argument('--init-variance', type=float, default=1.0,
                            help='Initialization value for variance')
        parser.add_argument("--input-feat-per-channel", type=int, default=80,
            metavar="N", help="encoder input dimension per input channel")
        parser.add_argument('--ctc-compress-strategy', type=str, default="none",
                            choices=['none', 'avg', 'weighted', 'softmax'],
                            help="Strategy to use when compressing CTC output")
        parser.add_argument('--freeze-pretrained', action='store_true',
                            help='if set, all params loaded from the pretrained model are freezed')

    @classmethod
    def build_encoder(cls, args, dictionary):
        encoder = S2TTransformerEncoderConv2d(args, dictionary)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
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
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, return_all_hiddens=True)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if self.encoder.ctc_flag:
            return decoder_out, {"ctc_out": encoder_out["ctc_out"], "ctc_lengths": encoder_out["ctc_lengths"]}
        else:
            return decoder_out


class S2TTransformerEncoderConv2d(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        stride = 2
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        kernel_size = [int(k) for k in args.conv_kernel_sizes.split(",")]
        self.subsample = Conv2dSubsampler(
            self.activation_fn,
            args.dropout,
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            kernel_size,
            stride,
        )

        if args.distance_penalty == True:
            args.distance_penalty = 'log'

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [ConvTransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

        self.num_layers = len(self.transformer_layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(args.encoder_embed_dim)
        else:
            self.layernorm_embedding = None

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        # ctc
        self.ctc_flag = False
        if args.criterion == "ctc_multi_loss" or args.ctc_compress_strategy != "none":
            self.ctc_flag = True
        if self.ctc_flag:
            self.ctc_fc = nn.Linear(args.encoder_embed_dim, len(dictionary))
            self.ctc_layer = args.ctc_encoder_layer
            if args.ctc_compress_strategy != "none":
                self.ctc_compress_method = getattr(CTCCompressStrategy, args.ctc_compress_strategy)
            else:
                self.ctc_compress_method = "none"


    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **kwargs):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        x_ctc = None
        for l_idx, layer in enumerate(self.transformer_layers):
            x = layer(x, encoder_padding_mask)
            # ctc
            if self.ctc_flag and self.ctc_layer == l_idx + 1:
                x_ctc = self.ctc_fc(x)
                if self.ctc_compress_method != "none":
                    x, input_lengths = self.average_same_ctc_features(x_ctc, x, input_lengths)
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        if self.ctc_flag:
            return {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [encoder_padding_mask],  # B x T
                "encoder_embedding": None,
                "encoder_states": encoder_states,  # List[T x B x C]
                "ctc_out": x_ctc,  # T x B x D
                "ctc_lengths": input_lengths
            }
        else:
            return {
                "encoder_out": [x],
                "encoder_padding_mask": [encoder_padding_mask],
                "encoder_embedding":None,
                "encoder_states": encoder_states,
                "src_tokens":None,
                "src_lengths":None,
            }

    def average_same_ctc_features(self, x_ctc, x, src_lengths):
        with torch.no_grad():
            batch_predicted = []
            prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
            for b in range(prob_ctc.shape[0]):
                predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])
            new_lengths = [len(p) for p in batch_predicted]
            weights_matrix = self.ctc_compress_method(prob_ctc, batch_predicted, new_lengths, x.dtype, x.device)
        # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
        compressed_output = x.permute(1, 2, 0).bmm(weights_matrix)  # B x C x T'
        return compressed_output.permute(2, 0, 1), src_lengths.new(new_lengths)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        # ctc
        if self.ctc_flag:
            new_ctc_out = encoder_out["ctc_out"].index_select(1, new_order)
            new_ctc_lengths = encoder_out["ctc_lengths"].index_select(0, new_order)
            return {
                "encoder_out": new_encoder_out,  # T x B x C
                "encoder_padding_mask": new_encoder_padding_mask,  # B x T
                "encoder_embedding": None,
                "encoder_states": None,  # List[T x B x C]
                "ctc_out": new_ctc_out,  # T x B x D
                "ctc_lengths": new_ctc_lengths
            }
        else:
            return {
                "encoder_out": new_encoder_out,  # T x B x C
                "encoder_padding_mask": new_encoder_padding_mask,  # B x T
                "encoder_embedding": None,  # B x T x C
                "encoder_states": None,  # List[T x B x C]
                "src_tokens": None,  # B x T
                "src_lengths": None,  # B x 1
            }

def BatchNorm(embedding_dim):
    m = nn.BatchNorm2d(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m

class CTCCompressStrategy:
    @staticmethod
    def avg(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = 1.0 / same[1]
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix.to(device)

    @staticmethod
    def weighted(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix

    @staticmethod
    def softmax(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = F.softmax(prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]])
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix

@register_model_architecture(model_name="s2t_transformer_conv2d", arch_name="s2t_transformer_conv2d")
def base_architecture(args):
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 64)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 768)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.dropout = getattr(args, "dropout", 0.3)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
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
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 256)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.distance_penalty = getattr(args, 'distance_penalty', False)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model_architecture("s2t_transformer_conv2d", "s2t_transformer_conv2d_big")
def s2t_transformer_conv2d_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.distance_penalty = getattr(args, 'distance_penalty', False)
    base_architecture(args)


@register_model_architecture("s2t_transformer_conv2d", "s2t_transformer_conv2d_big2")
def s2t_transformer_conv2d_big2(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    s2t_transformer_conv2d_big(args)


@register_model_architecture("s2t_transformer_conv2d", "s2t_transformer_conv2d_giant")
def s2t_transformer_conv2d_giant(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.conv_channels = getattr(args, "conv_channels", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1024)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    s2t_transformer_conv2d_big(args)