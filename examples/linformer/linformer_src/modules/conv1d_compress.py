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

from functools import reduce

import torch.nn as nn
import math


class Conv1dCompressLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            kernel_size,
            compression_factor: int,
            padding: int,
            n_layers: int,
            freeze_compress: False,
    ):
        super(Conv1dCompressLayer, self).__init__()
        # compute the stride for each layer to obtain the compression factor at the output of the Conv1d stack
        stride_per_layer = compression_factor // n_layers
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels=embed_dim if i == 0 else embed_dim // 2,
                out_channels=embed_dim if i < n_layers - 1 else embed_dim * 2,
                kernel_size=kernel_size,
                stride=stride_per_layer,
                padding=padding,
            )
            for i in range(n_layers)
        )

        # Check if the final stride of the Convolutional layers corresponds to the compression factor
        strides = [layer.stride[0] for layer in self.conv_layers]
        assert reduce(lambda x, y: x*y, strides) == compression_factor

        for cl in self.conv_layers:
            # intialize parameters for compressed layer
            nn.init.xavier_uniform_(cl.weight, gain=1 / math.sqrt(2))

            if freeze_compress:
                cl.weight.requires_grad = False

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        return x
