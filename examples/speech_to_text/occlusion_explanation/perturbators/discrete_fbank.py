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

import logging
from typing import Tuple

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.perturbators import register_perturbator
from examples.speech_to_text.occlusion_explanation.perturbators.continuous_fbank import \
    ContinuousOcclusionFbankPerturbator


LOGGER = logging.getLogger(__name__)


@register_perturbator("discrete_fbank_time")
class DiscreteTimeOcclusionFbankPerturbator(ContinuousOcclusionFbankPerturbator):
    """
    Class for implementing occlusion perturbations discrete in the time domain.
    In this method, entire time frames in the input data are perturbed.
    """
    def __call__(self, fbank: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        mask = (torch.rand(fbank.size(0)) > self.mask_probability).to(fbank.dtype)
        masked_fbank = fbank * mask.unsqueeze(1)
        return mask, masked_fbank


@register_perturbator("discrete_fbank_frequency")
class DiscreteFrequencyOcclusionFbankPerturbator(ContinuousOcclusionFbankPerturbator):
    """
    Class for implementing occlusion perturbations discrete in the frequency domain.
    In this method, entire frequency rows in the input data are perturbed.
    """
    def __call__(self, fbank: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        mask = (torch.rand(fbank.size(1)) > self.mask_probability).to(fbank.dtype)
        masked_fbank = fbank * mask.unsqueeze(0)
        return mask, masked_fbank
