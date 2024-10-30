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


@register_normalizer("single_mean_std")
class SingleMeanStdNormalizer(Normalizer):
    """
    Perform mean-standard deviation normalization of fbanks and
    tgt map separately map at token level.
    """
    def __call__(
            self, fbank_explanation: Tensor, tgt_explanation: Tensor) -> Tuple[Tensor, Tensor]:
        tokens_size = fbank_explanation.shape[0]

        # fbank
        flattened_fbank = fbank_explanation.view(tokens_size, -1)
        fbank_row_wise_count = flattened_fbank.shape[1]
        fbank_expected_values = flattened_fbank.sum(dim=1) / fbank_row_wise_count
        fbank_expected_squared_values = (flattened_fbank ** 2).sum(dim=1) / fbank_row_wise_count
        momentum_order_2 = fbank_expected_squared_values - fbank_expected_values ** 2
        # adjustment to compute the sampling variance
        fbank_std_devs = torch.sqrt(momentum_order_2 * fbank_row_wise_count / (fbank_row_wise_count - 1))
        fbank_expected_values = fbank_expected_values.unsqueeze(-1).unsqueeze(-1)
        fbank_std_devs = fbank_std_devs.unsqueeze(-1).unsqueeze(-1)
        fbank_map_norm = (fbank_explanation - fbank_expected_values) / fbank_std_devs
        # set value for problematic cases (e.g. when std_devs is 0 and the score becomes inf)
        problematic_mask = torch.isnan(fbank_map_norm) | torch.isinf(fbank_map_norm)
        fbank_map_norm[problematic_mask] = 0.

        # tgt
        tgt_padding_mask = torch.ones(
            tgt_explanation.shape[:2], device=tgt_explanation.device).tril().unsqueeze(-1) != 0
        padded_tgt_map = torch.where(
            tgt_padding_mask,
            tgt_explanation,
            torch.tensor(0.0, dtype=tgt_explanation.dtype)).view(tokens_size, -1)
        tgt_row_wise_count = tgt_padding_mask.squeeze(-1).sum(dim=1)
        tgt_expected_values = padded_tgt_map.sum(dim=1) / tgt_row_wise_count
        tgt_expected_squared_values = (padded_tgt_map ** 2).sum(dim=1) / tgt_row_wise_count
        momentum_order_2 = tgt_expected_squared_values - tgt_expected_values ** 2
        # adjustment to compute the sampling variance
        tgt_std_devs = torch.sqrt(momentum_order_2 * tgt_row_wise_count / (tgt_row_wise_count - 1))
        tgt_expected_values = tgt_expected_values.unsqueeze(-1).unsqueeze(-1)
        tgt_std_devs = tgt_std_devs.unsqueeze(-1).unsqueeze(-1)
        tgt_map_norm = (tgt_explanation - tgt_expected_values) / tgt_std_devs
        tgt_map_norm = torch.where(
            tgt_padding_mask, tgt_map_norm, torch.tensor(0.0, dtype=tgt_explanation.dtype))
        # set value for first step where there is only one token
        # and also for other problematic cases (e.g. when std_devs is 0 and the score becomes inf)
        problematic_mask = torch.isnan(tgt_map_norm) | torch.isinf(tgt_map_norm)
        tgt_map_norm[problematic_mask] = 0.

        return fbank_map_norm, tgt_map_norm
