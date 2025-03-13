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

from examples.speech_to_text.occlusion_explanation.normalizers import Normalizer, register_normalizer
from examples.speech_to_text.occlusion_explanation.normalizers.paired_min_max import PairedMinMaxNormalizer


@register_normalizer("gender_paired_min_max")
class PairedMinMaxNormalizerGender(PairedMinMaxNormalizer):
    """
    Perform min-max normalization across fbank and previous output tokens map.
    This normalizer expects a single explanation for a single term instead of one explanation
    per token in the generated target sequence.
    """
    def __call__(
            self,
            fbank_explanation: Tensor,
            tgt_explanation: Tensor,
            gender_term_index: int) -> Tuple[Tensor, Tensor]:
        assert fbank_explanation.shape[0] == 1, \
            "gender_paired_min_max normalizer expects a single explanation per sample, " \
            f"but got {fbank_explanation.shape[0]}."
        
        # Create a mask to ignore tokens after the gender term in the target explanation
        tgt_padding_mask = torch.ones(
            tgt_explanation.shape[:2], device=tgt_explanation.device)
        tgt_padding_mask[:, gender_term_index + 1:] = 0.0   # +1 because of the BOS token
        tgt_padding_mask = tgt_padding_mask.unsqueeze(-1) != 0  # boolean tensor (1, tgt_len, 1)

        return self.apply_min_max_normalization(fbank_explanation, tgt_explanation, tgt_padding_mask)
