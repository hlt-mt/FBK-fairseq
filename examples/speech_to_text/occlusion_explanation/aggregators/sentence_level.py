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
from typing import Tuple, Dict, Union, List

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.aggregators import \
    register_aggregator, Aggregator


class SentenceLevelAggregator(Aggregator):

    def _normalize(self, fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        """
        It gets single explanations for both the filterbank and the previous output tokens
        and returns them normalized.
        """
        raise NotImplementedError

    def __call__(
            self, explanations: Dict[int, Dict[str, Union[Tensor, str, List[str]]]]
    ) -> Dict[int, Tuple[Tensor, Tensor]]:
        """
        Aggregates (by summing them) the heatmaps of fbank and token embeddings
        over all the tokens in a sample.
        """
        aggregated_explanations = {}
        for i, explanation in explanations.items():
            fbank = explanation["fbank_heatmap"]
            tgt = explanation["tgt_embed_heatmap"]
            fbank, tgt = self._normalize(fbank, tgt)
            aggregated_explanations[i] = fbank.sum(dim=0), tgt.sum(dim=0)
        return aggregated_explanations


@register_aggregator("sentence_aggregator_no_norm")
class SentenceLevelAggregatorNoNormalization(SentenceLevelAggregator):
    def _normalize(self, fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Identity function, does not perform any normalization.
        """
        return fbank_map, tgt_map


@register_aggregator("sentence_aggregator_min_max_norm")
class SentenceLevelAggregatorMinMaxNormalization(SentenceLevelAggregator):
    def _normalize(self, fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform min-max normalization across fbank and previous output
        tokens map related to a single token along the first dimension.
        """
        fbank_map_norm = torch.zeros_like(fbank_map, dtype=torch.float32)
        tgt_map_norm = torch.zeros_like(tgt_map, dtype=torch.float32)
        for i in range(fbank_map.shape[0]):
            min_val = torch.min(torch.min(fbank_map[i]), torch.min(tgt_map[i]))
            max_val = torch.max(torch.max(fbank_map[i]), torch.max(tgt_map[i]))
            values_range = max_val - min_val
            fbank_map_norm[i] = (fbank_map[i] - min_val) / values_range
            tgt_map_norm[i] = (tgt_map[i] - min_val) / values_range
        return fbank_map_norm, tgt_map_norm


@register_aggregator("sentence_aggregator_mean_std_norm")
class SentenceLevelAggregatorMeanStdNormalization(SentenceLevelAggregator):
    def _normalize(self, fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform mean-standard deviation normalization across fbank and previous output
        tokens map related to a single token along the first dimension.
        """
        fbank_map_norm = torch.zeros_like(fbank_map)
        tgt_map_norm = torch.zeros_like(tgt_map)
        for i in range(fbank_map.shape[0]):
            mean = torch.mean(torch.cat([fbank_map[i].view(-1), tgt_map[i].view(-1)]), dim=0)
            std = torch.std(torch.cat([fbank_map[i].view(-1), tgt_map[i].view(-1)]), dim=0)
            fbank_map_norm[i] = (fbank_map[i] - mean) / std
            tgt_map_norm[i] = (tgt_map[i] - mean) / std
        return fbank_map_norm, tgt_map_norm
