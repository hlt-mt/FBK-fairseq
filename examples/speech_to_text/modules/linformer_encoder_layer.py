# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from examples.linformer.linformer_src.modules.conv_multihead_linear_attention import ConvMultiheadLinearAttention
from fairseq.modules import TransformerEncoderLayer


class LinformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, shared_compress_layer):
        # wrap in a list so it's not automatically registered by PyTorch
        self.shared_compress_layer = [shared_compress_layer]
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return ConvMultiheadLinearAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.dropout,
            self_attention=True,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            compressed=args.compressed,
            max_seq_len=args.max_source_positions,
            shared_kv_compressed=args.shared_kv_compressed,
            shared_compress_layer=self.shared_compress_layer[0],
            freeze_compress=args.freeze_compress,
            compress_kernel_size=args.compress_kernel_size,
        )
