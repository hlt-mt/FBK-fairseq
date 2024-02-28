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
from typing import Dict, Tuple

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.perturbators import \
    register_perturbator, OcclusionFbankPerturbator


LOGGER = logging.getLogger(__name__)


@register_perturbator("continuous_fbank")
class ContinuousOcclusionFbankPerturbator(OcclusionFbankPerturbator):
    """
    Class for implementing continuous occlusion perturbations.
    Under this method, each value in the input data is perturbed independently of the others.
    """
    def __init__(self, mask_probability: float, n_masks: int):
        self.mask_probability = mask_probability
        LOGGER.info(f"Masking fbanks with probability of {self.mask_probability}.")
        self.n_masks = n_masks
        LOGGER.info(f"{self.n_masks} masks are used.")

    @classmethod
    def from_config_dict(cls, config: Dict = None):
        if config is None:
            config = {}
        fbank_occlusion = config.get("fbank_occlusion", {})
        mask_probability = fbank_occlusion.get("p", 0.5)
        assert 0. < mask_probability <= 1.
        n_masks = fbank_occlusion.get("n_masks", 8000)
        return cls(mask_probability=mask_probability, n_masks=n_masks)

    def __call__(self, fbank: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Builds mask and performs masking for filterbank fed into the encoder.
        Args:
            - fbank: Tensor: a single filterbank of shape (time, channels)
        Returns:
            - mask: Tensor of shape (time, channels)
            - masked_fbank: masked version of the original fbank with shape (time, channels)
        """
        mask = (torch.rand(fbank.size()) > self.mask_probability).to(fbank.dtype)
        masked_fbank = fbank * mask
        return mask, masked_fbank
