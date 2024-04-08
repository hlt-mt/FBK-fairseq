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
from typing import Dict, Union, List

from torch import Tensor

from examples.speech_to_text.occlusion_explanation.aggregators import \
    register_aggregator, Aggregator


@register_aggregator("frequency_embed")
class FrequencyEmbeddingAggregator(Aggregator):
    """
    Aggregates explanations by collapsing the channel/embedding dimension,
    taking the mean along that dimension (dimension 2) for filterbanks and
    target embeddings, respectively.
    For filterbanks, this operation transforms the size from (seq_len, time,
    channels) to (seq_len, time, 1), preserving the temporal structure while
    reducing the channel dimension to a single value.
    For target embeddings, this operation transforms the size from (seq_len,
    seq_len, embed_dim) to (seq_len, seq_len, 1), preserving the sequence
    structure while reducing th embedding dimension to a single value.
    If the original explanations are already discrete with size (seq_len,
    time/seq_len 1), they remain unchanged.
    """
    def __call__(
            self, explanations: Dict[int, Dict[str, Union[Tensor, str, List[str]]]]
    ) -> Dict[int, Dict[str, Union[Tensor, List[str]]]]:
        """
        Performs aggregation.
        """
        for sample_id in explanations.keys():
            fbank = explanations[sample_id]["fbank_heatmap"]
            tgt = explanations[sample_id]["tgt_embed_heatmap"]
            explanations[sample_id]["fbank_heatmap"] = fbank.mean(dim=2, keepdim=True)
            explanations[sample_id]["tgt_embed_heatmap"] = tgt.mean(dim=2, keepdim=True)
        return explanations
