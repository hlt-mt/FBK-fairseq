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
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from examples.speech_to_text.modules.speechformer_encoder_layer import SpeechformerEncoderLayer
from fairseq import checkpoint_utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable, Conv1dSubsampler, \
    S2TTransformerModel
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
            '--ctc-compress-strategy', type=str, default="none",
            choices=['none', 'avg', 'weighted', 'softmax'],
            help="strategy to use when compressing the CTC output"
        )
        parser.add_argument(
            '--ctc-compress-max-out-size', type=int, default=-1,
            help="if CTC compression is enabled and this argument is set to a positive number, "
                 "every input is forced to be at most as long as the value set for this parameter, "
                 "even though the CTC would not compress it enough. Intuitively, this parameter "
                 "should be set to 1/4 of the max input length to ensure that the maximum sequence "
                 "length of the Transformer self-attention input is the same as in the case of "
                 "of models having 2 initial convolutions with stride 2."
        )
        parser.add_argument(
            '--add-position-after-ctc', action='store_true',
            help="Add positional embedding after CTC compression"
        )
        parser.add_argument(
            '--transformer-after-compression', default=False, action='store_true',
            help='whether or not using standard TransformerEncoder layers after CTC compression '
            'instead of ConvAttention Encoder layers')
        parser.add_argument(
            '--allow-partial-encoder-loading', action='store_true', default=False,
            help="if set, the model is restored even if it doesn't match exactly"
            "the architecture, ie. some params are missing."
        )
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
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from,
                allow_partial_encoder_loading=getattr(args, "allow_partial_encoder_loading", False),
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


class SpeechformerEncoder(FairseqEncoder):
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
        if args.transformer_after_compression:
            self.speechformer_layers = nn.ModuleList(
                [self.build_speechformer_encoder_layer(args) for _ in range(args.ctc_encoder_layer)]
            )
            self.speechformer_layers.extend(
                [TransformerEncoderLayer(args) for _ in range(args.encoder_layers - args.ctc_encoder_layer)]
            )
        else:
            self.speechformer_layers = nn.ModuleList(
                [self.build_speechformer_encoder_layer(args) for _ in range(args.encoder_layers)]
            )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        # ctc
        self.ctc_flag = False
        if args.criterion == "ctc_multi_loss" or args.ctc_compress_strategy != "none":
            self.ctc_flag = True
        if self.ctc_flag:
            self.ctc_fc = nn.Linear(args.encoder_embed_dim, len(dictionary))
            self.ctc_layer = args.ctc_encoder_layer
            if args.ctc_compress_strategy != "none":
                self.ctc_compress_method = getattr(CTCCompressStrategy, args.ctc_compress_strategy)
                self.ctc_compress_max_out_size = args.ctc_compress_max_out_size
            else:
                self.ctc_compress_method = "none"
            self.ctc_compress_add_pos = args.add_position_after_ctc

    def build_speechformer_encoder_layer(self, args):
        if args.shared_layer_kv_compressed == 1 and self.compress_layer is None:
            compress_layer = nn.Conv1d(
                args.encoder_embed_dim,
                args.encoder_embed_dim,
                args.compress_kernel_size,
                stride=args.compressed,
                padding=args.compress_kernel_size // 2,
            )
            # intialize parameters for compressed layer
            nn.init.xavier_uniform_(compress_layer.weight, gain=1 / math.sqrt(2))
            if args.freeze_compress == 1:
                compress_layer.weight.requires_grad = False
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
                x_ctc = self.ctc_fc(x)
                if self.ctc_compress_method != "none":
                    x, input_lengths = self.average_same_ctc_features(x_ctc, x, input_lengths)
                    encoder_padding_mask = lengths_to_padding_mask(input_lengths)
                    if self.ctc_compress_add_pos:
                        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
                        x += positions
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.ctc_flag:
            return {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
                "encoder_embedding": [],
                "encoder_states": encoder_states,  # List[T x B x C]
                "ctc_out": x_ctc,  # T x B x D
                "ctc_lengths": ctc_lengths,
                "src_tokens": [],
                "src_lengths": [],
            }
        else:
            return {
                "encoder_out": [x],
                "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],
                "encoder_embedding": [],
                "encoder_states": encoder_states,
                "src_tokens": [],
                "src_lengths": [],
            }

    def ensure_max_ctc_out_len(self, batch_predicted):
        """
        Ensures that the output of the CTC compression is not longer than the ctc_compress_max_out_size.
        If there are samples violating this constraints, consecutive predictions are merged
        so to shorten the sentence.
        E.g. if the ctc_compress_max_out_size is set to 3, and the output of the CTC compression would be
        long 5, the first and second predictions are merged, as well as the third and the fourth. So, the
        corresponding vectors will be merged according to the CTC compression strategy.
        """
        if self.ctc_compress_max_out_size > 0:

            def merge_sublist(elements):
                """
                Takes a list of Tuples (predicted_element, num_corresponding_vectors) and returns
                a single tuple with the predicted_element having the highest number of corresponding_vectors
                (in case of a tie, the first is returned) and the total sum of the num_corresponding_vectors
                E.g. if the input is [(a, 3), (b, 5), (c, 6), (a, 4)], the output will be (a, 18).
                """
                sum_num_vectors = 0
                max_element = None
                max_element_cnt = 0
                temp_dict = {}
                for predicted_element, num_corresponding_vectors in elements:
                    if predicted_element in temp_dict:
                        temp_dict[predicted_element] += num_corresponding_vectors
                    else:
                        temp_dict[predicted_element] = num_corresponding_vectors
                    if temp_dict[predicted_element] > max_element_cnt:
                        max_element_cnt = temp_dict[predicted_element]
                        max_element = predicted_element
                    sum_num_vectors += num_corresponding_vectors
                return max_element, sum_num_vectors

            for b_idx, p in enumerate(batch_predicted):
                pred_len = len(p)
                if pred_len > self.ctc_compress_max_out_size:
                    reduction_factor = math.ceil(pred_len / self.ctc_compress_max_out_size)
                    i = 0
                    new_p = []
                    while i < pred_len:
                        new_p.append(merge_sublist(p[i:i+reduction_factor]))
                        i += reduction_factor
                    batch_predicted[b_idx] = new_p

        return batch_predicted

    def average_same_ctc_features(self, x_ctc, x, src_lengths):
        with torch.no_grad():
            batch_predicted = []
            prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
            for b in range(prob_ctc.shape[0]):
                predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])
            batch_predicted = self.ensure_max_ctc_out_len(batch_predicted)
            new_lengths = [len(p) for p in batch_predicted]
            weights_matrix = self.ctc_compress_method(prob_ctc, batch_predicted, new_lengths, x.dtype, x.device)
        # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
        compressed_output = x.permute(1, 2, 0).bmm(weights_matrix)  # B x C x T'
        return compressed_output.permute(2, 0, 1), src_lengths.new(new_lengths)

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to *new_order*.

            Args:
                encoder_out: output from the ``forward()`` method
                new_order (LongTensor): desired order

            Returns:
                *encoder_out* rearranged according to *new_order*

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

        # ctc
        if self.ctc_flag:
            new_ctc_out = encoder_out["ctc_out"].index_select(1, new_order)
            new_ctc_lengths = encoder_out["ctc_lengths"].index_select(0, new_order)

            return {
                "encoder_out": new_encoder_out,  # T x B x C
                "encoder_padding_mask": new_encoder_padding_mask,  # B x T
                "encoder_embedding": new_encoder_embedding,  # B x T x C
                "encoder_states": encoder_states,  # List[T x B x C]
                "src_tokens": [],  # B x T
                "src_lengths": [],  # B x 1
                "ctc_out": new_ctc_out,  # T x B x D
                "ctc_lengths": new_ctc_lengths,
            }
        else:
            return {
                "encoder_out": new_encoder_out,  # T x B x C
                "encoder_padding_mask": new_encoder_padding_mask,  # B x T
                "encoder_embedding": new_encoder_embedding,  # B x T x C
                "encoder_states": encoder_states,  # List[T x B x C]
                "src_tokens": [],  # B x T
                "src_lengths": [],  # B x 1
            }


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

