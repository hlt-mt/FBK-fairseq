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


def _min_max_normalization(fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Perform min-max normalization across fbank and previous output
    tokens map related to a single token along the first dimension.
    """
    tokens_size = fbank_map.shape[0]
    upper_left_tr = torch.ones(tgt_map.shape[:2], device=tgt_map.device).tril().unsqueeze(-1) != 0
    min_vals = torch.minimum(
        fbank_map.view(tokens_size, -1).min(dim=1).values,
        torch.where(upper_left_tr, tgt_map, torch.inf).view(tokens_size, -1).min(dim=1).values)
    max_vals = torch.maximum(
        fbank_map.view(tokens_size, -1).max(dim=1).values,
        torch.where(upper_left_tr, tgt_map, -torch.inf).view(tokens_size, -1).max(dim=1).values)
    min_vals = min_vals.unsqueeze(-1).unsqueeze(-1)
    max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)
    values_ranges = max_vals - min_vals
    fbank_map_norm = (fbank_map - min_vals) / values_ranges
    tgt_map_norm = (tgt_map - min_vals) / values_ranges
    tgt_map_norm = torch.where(upper_left_tr, tgt_map_norm, 0.)
    return fbank_map_norm, tgt_map_norm


def _mean_std_normalization(fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Perform mean-standard deviation normalization across fbank and previous output
    tokens map related to a single token along the first dimension.
    """
    tokens_size = fbank_map.shape[0]
    flattened_fbank = fbank_map.view(tokens_size, -1)
    upper_left_tr = torch.ones(tgt_map.shape[:2], device=tgt_map.device).tril().unsqueeze(-1) != 0
    padded_tgt_map = torch.where(upper_left_tr, tgt_map, 0.0).view(tokens_size, -1)
    # we compute the standard deviation using the following theorem:
    # var[X] = E[X^2] - (E[X])^2
    # where E[X] is the expected value (i.e. the mean) of the casual variable X
    row_wise_count = upper_left_tr.squeeze(-1).sum(dim=1) + flattened_fbank.shape[1]
    expected_values = (flattened_fbank.sum(dim=1) + padded_tgt_map.sum(dim=1)) / row_wise_count
    expected_squared_values = (
        (flattened_fbank ** 2).sum(dim=1) + (padded_tgt_map ** 2).sum(dim=1)) / row_wise_count
    std_devs = torch.sqrt(expected_squared_values - expected_values ** 2)
    expected_values = expected_values.unsqueeze(-1).unsqueeze(-1)
    std_devs = std_devs.unsqueeze(-1).unsqueeze(-1)
    fbank_map_norm = (fbank_map - expected_values) / std_devs
    tgt_map_norm = (tgt_map - expected_values) / std_devs
    tgt_map_norm = torch.where(upper_left_tr, tgt_map_norm, 0.)
    return fbank_map_norm, tgt_map_norm
