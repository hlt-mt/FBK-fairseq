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
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor

from fairseq.modules import FairseqDropout


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout

    Inputs: inputs
        x (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.layernorm = nn.LayerNorm(encoder_dim)
        self.dropout_module = FairseqDropout(p=dropout_p, module_name=self.__class__.__name__)
        self.first_linear = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        init.xavier_uniform_(self.first_linear.weight)
        init.zeros_(self.first_linear.bias)
        self.second_linear = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        init.xavier_uniform_(self.second_linear.weight)
        init.zeros_(self.second_linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layernorm(x)
        x = self.first_linear(x)
        x = F.silu(x)
        x = self.dropout_module(x)
        x = self.second_linear(x)
        x = self.dropout_module(x)
        return x
