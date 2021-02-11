# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from examples.linformer.linformer_src.modules.multihead_linear_attention import MultiheadLinearAttention
from fairseq.modules import TransformerEncoderLayer

SHARED_COMPRESS_LAYER = None


class LinformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
            self,
            args,
            compressed: int = 1,
            max_seq_len: int = 512,
            shared_kv_compressed: int = 0,
            shared_compress_layer: any = None,
            freeze_compress: int = 0,
    ) -> None:
        # Initialize linformer parameters
        self.compressed = compressed
        self.max_seq_len = max_seq_len
        self.shared_kv_compressed = shared_kv_compressed
        self.freeze_compress = freeze_compress
        super().__init__(args)
        global SHARED_COMPRESS_LAYER

        SHARED_COMPRESS_LAYER = shared_compress_layer

    def build_self_attention(self, embed_dim, args):
        global SHARED_COMPRESS_LAYER
        return MultiheadLinearAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            compressed=self.compressed,
            max_seq_len=self.max_seq_len,
            shared_kv_compressed=self.shared_kv_compressed,
            shared_compress_layer=SHARED_COMPRESS_LAYER,
            freeze_compress=self.freeze_compress,
        )
