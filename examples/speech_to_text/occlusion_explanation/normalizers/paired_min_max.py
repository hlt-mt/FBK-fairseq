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


@register_normalizer("paired_min_max")
class PairedMinMaxNormalizer(Normalizer):
    """
    Perform min-max normalization across fbank and previous output
    tokens map at token level.
    """
    def __call__(
            self, fbank_explanation: Tensor, tgt_explanation: Tensor) -> Tuple[Tensor, Tensor]:
        tokens_size = fbank_explanation.shape[0]
        tgt_padding_mask = torch.ones(
            tgt_explanation.shape[:2], device=tgt_explanation.device).tril().unsqueeze(-1) != 0
        min_vals = torch.minimum(
            fbank_explanation.view(tokens_size, -1).min(dim=1).values,
            torch.where(
                tgt_padding_mask,
                tgt_explanation,
                torch.tensor(torch.inf, device=tgt_explanation.device)).view(tokens_size, -1).min(dim=1).values)
        max_vals = torch.maximum(
            fbank_explanation.view(tokens_size, -1).max(dim=1).values,
            torch.where(
                tgt_padding_mask,
                tgt_explanation,
                torch.tensor(-torch.inf, device=tgt_explanation.device)).view(tokens_size, -1).max(dim=1).values)
        min_vals = min_vals.unsqueeze(-1).unsqueeze(-1)
        max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)
        values_ranges = max_vals - min_vals
        fbank_map_norm = (fbank_explanation - min_vals) / values_ranges
        tgt_map_norm = (tgt_explanation - min_vals) / values_ranges
        tgt_map_norm = torch.where(
            tgt_padding_mask, tgt_map_norm, torch.tensor(0.0, dtype=tgt_explanation.dtype))
        return fbank_map_norm, tgt_map_norm
