# Copyright 2025 FBK

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


@register_normalizer("gender_single_min_max")
class SingleMinMaxNormalizerGender(Normalizer):
    """
    Perform min-max normalization of the fbank and previous output
    tokens maps, separately. This version of the normalizer expects 
    a single explanation for a single term instead of one explanation
    per token in the generated target sequence.
    """
    def __call__(
            self,
            fbank_explanation: Tensor,
            tgt_explanation: Tensor,
            gender_term_index: int) -> Tuple[Tensor, Tensor]:
        n_explanations = fbank_explanation.shape[0]
        assert n_explanations == 1, \
            f"gender_paired_min_max normalizer expects a single explanation per sample, but got {n_explanations}."
        
        tgt_padding_mask = torch.ones(
            tgt_explanation.shape[:2], device=tgt_explanation.device)
        tgt_padding_mask[:, gender_term_index + 1:] = 0.0  # +1 because of the BOS token
        tgt_padding_mask = tgt_padding_mask.unsqueeze(-1) != 0  # boolean tensor (1, tgt_len, 1)
        
        # Filterbank saliency map normalization
        min_vals = fbank_explanation.view(n_explanations, -1).min(dim=1).values
        max_vals = fbank_explanation.view(n_explanations, -1).max(dim=1).values
        min_vals = min_vals.unsqueeze(-1).unsqueeze(-1)
        max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)
        values_ranges = max_vals - min_vals
        fbank_map_norm = (fbank_explanation - min_vals) / values_ranges
        
        # Previous output tokens saliency map normalization
        min_vals = torch.where(
            tgt_padding_mask,
            tgt_explanation,
            torch.tensor(torch.inf, device=tgt_explanation.device)).view(n_explanations, -1).min(dim=1).values
        max_vals = torch.where(
            tgt_padding_mask,
            tgt_explanation,
            torch.tensor(-torch.inf, device=tgt_explanation.device)).view(n_explanations, -1).max(dim=1).values
        min_vals = min_vals.unsqueeze(-1).unsqueeze(-1)
        max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)
        values_ranges = max_vals - min_vals
        tgt_map_norm = (tgt_explanation - min_vals) / values_ranges
        tgt_map_norm = torch.where(
            tgt_padding_mask, tgt_map_norm, torch.tensor(0.0, dtype=tgt_explanation.dtype))
        
        return fbank_map_norm, tgt_map_norm
