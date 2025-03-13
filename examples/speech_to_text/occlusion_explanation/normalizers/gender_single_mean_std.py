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
from typing import Tuple

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.normalizers import register_normalizer
from examples.speech_to_text.occlusion_explanation.normalizers.single_mean_std import SingleMeanStdNormalizer


@register_normalizer("gender_single_mean_std")
class SingleMeanStdNormalizerGender(SingleMeanStdNormalizer):
    """
    Perform mean normalization of the speech input and target tokens
    separately. This normalizer expects a single explanation for a single term.
    """
    def __call__(
            self,
            fbank_explanation: Tensor,
            tgt_explanation: Tensor,
            gender_term_index: int) -> Tuple[Tensor, Tensor]:
        
        assert fbank_explanation.shape[0] == 1, \
            "gender_single_mean_std normalizer expects a single explanation per sample, " \
            f"but got {fbank_explanation.shape[0]}."

        tgt_ignore_mask = torch.ones(
            tgt_explanation.shape,
            device=tgt_explanation.device,
            dtype=torch.bool)
        tgt_ignore_mask[:, gender_term_index + 1:, :] = False

        return self.apply_single_mean_std_normalization(
            fbank_explanation, tgt_explanation, tgt_ignore_mask)
