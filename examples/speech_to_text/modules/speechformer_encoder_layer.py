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

from examples.linformer.linformer_src.modules.conv_multihead_linear_attention import ConvAttention
from fairseq.modules import TransformerEncoderLayer


class SpeechformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, shared_compress_layer):
        # wrap in a list so it's not automatically registered by PyTorch
        self.shared_compress_layer = [shared_compress_layer]
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return ConvAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.dropout,
            self_attention=True,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            compressed=args.compressed,
            shared_kv_compressed=args.shared_kv_compressed,
            shared_compress_layer=self.shared_compress_layer[0],
            freeze_compress=args.freeze_compress,
            compress_kernel_size=args.compress_kernel_size,
        )
