# This code was inspired to the Soohwan Kim (https://github.com/sooftware/conformer) repository.
# Read carefully their licence.

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

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.modules import FairseqDropout


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with the first pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models. Then, Swift (or SiLu) activation function is applied and followed by the second
    pointwise convolution. The Dropout module is applied in the end.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        x (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        **outputs** (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, only supports expansion_factor 2"
        self.layernorm = nn.LayerNorm(in_channels)
        self.batchnorm = nn.SyncBatchNorm(in_channels)
        self.first_pointwise_conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * expansion_factor,
            kernel_size=(1, ),
            stride=(1, ),
            padding=0,
            bias=True,
        )
        self.second_pointwise_conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, ),
            stride=(1, ),
            padding=0,
            bias=True,
        )
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, ),
            stride=(1, ),
            groups=in_channels,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.dropout_module = FairseqDropout(p=dropout_p, module_name=self.__class__.__name__)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layernorm(x).transpose(1, 2)
        x = self.first_pointwise_conv1d(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv1d(x)
        x = self.batchnorm(x)
        x = F.silu(x)
        x = self.second_pointwise_conv1d(x)
        x = self.dropout_module(x)
        return x.transpose(1, 2)
