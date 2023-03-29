# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import math
from typing import Any

from torch import nn, tensor
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: Any, x, lambda_factor) -> Any:
        ctx.save_for_backward(lambda_factor)
        return x

    @staticmethod
    def backward(ctx: Any, grad_output) -> Any:
        lambda_factor, = ctx.saved_tensors
        return grad_output.neg() * lambda_factor, None


gradient_reversal = GradientReversalFunction.apply


class GradientReversalLayer(nn.Module):
    def __init__(
            self,
            lambda_factor=None,
            max_updates=None,
            gamma=None,
            *args, **kwargs):
        """
        A gradient reversal layer that has a gradient scaling parameter
        as per "Unsupervised Domain Adaptation by Backpropagation"
        (https://arxiv.org/pdf/1409.7495.pdf).
        The gradient scaling parameter follows the updating schema defined in
        the paper, which is:

        2 / (1 + math.exp(-gamma * p)) - 1

        where p is the ratio between the current number of updates and the maximum
        number of updates (ie. the percentage of training advancement), and gamma is
        an hyperparam, which is set to 10 in the original paper.

        Alternatively, setting `lambda_factor` keeps fixed the gradient scaling parameter.
        """
        super().__init__(*args, **kwargs)
        self._max_updates = max_updates
        self._adaptive_factor = lambda_factor is None
        if self._adaptive_factor:
            self._lambda_factor = tensor(0., requires_grad=False)
            assert gamma is not None, "either lambda_factor or gamma should not be None"
            assert self._max_updates is not None and self._max_updates > 0, \
                "max_updates should be a positive number when adaptive factor is used"
            self._gamma = gamma
        else:
            self._lambda_factor = tensor(lambda_factor, requires_grad=False)

    def forward(self, input_):
        return gradient_reversal(input_, self._lambda_factor)

    def set_num_updates(self, num_updates):
        if self._adaptive_factor:
            p = num_updates / self._max_updates
            self._lambda_factor = tensor(2 / (1 + math.exp(-self._gamma * p)) - 1, requires_grad=False)
