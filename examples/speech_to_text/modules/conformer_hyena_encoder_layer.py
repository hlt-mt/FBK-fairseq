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
from examples.speech_to_text.modules.conformer_encoder_layer import ConformerEncoderLayer
from examples.speech_to_text.modules.hyena import HyenaOperator


class ConformerHyenaEncoderLayer(ConformerEncoderLayer):
    """
    Conformer block with Hyena module instead of the Multihead Self-Attention with
    relative positions.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        x (time, batch, dim): Tensor containing input vector

    Returns: outputs
        **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(self, args):
        super().__init__(args)

    def build_attention(self, args):
        return HyenaOperator(
            d_model=self.encoder_dim,
            l_max=args.max_source_positions,
            order=args.hyena_order,
            filter_order=args.hyena_filter_order,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout_p,
            filter_dropout=self.attention_dropout_p,
            causal=getattr(args, 'hyena_causal', False)
        )
