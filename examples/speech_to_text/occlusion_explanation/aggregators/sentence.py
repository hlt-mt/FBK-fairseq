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


@register_aggregator("sentence")
class SentenceLevelAggregator(Aggregator):
    """
    Aggregates the explanations of filterbanks and target embeddings over all the tokens
    in a sample by averaging them. This operation transforms a filterbank explanation with
    size (seq_len, time, channels) to a size of (1, time, channels), while a target embedding
    explanation with size (seq_len, seq_len, embed_dim) becomes (1, seq_len, embed_dim).
    """
    def __call__(
            self, explanations: Dict[int, Dict[str, Union[Tensor, str, List[str]]]]
    ) -> Dict[int, Dict[str, Union[Tensor, List[str]]]]:
        """
        Performs aggregation.
        """
        aggregated_explanations = {}
        for sample_id, explanation_data in explanations.items():
            # initialize with key and values of explanations, so that optional keys are copied
            aggregated_explanations[sample_id] = {k: v for k, v in explanations[sample_id].items()}
            fbank = explanation_data["fbank_heatmap"]
            tgt = explanation_data["tgt_embed_heatmap"]
            aggregated_explanations[sample_id]["fbank_heatmap"] = fbank.mean(dim=0, keepdim=True)
            aggregated_explanations[sample_id]["tgt_embed_heatmap"] = tgt.mean(dim=0, keepdim=True)
        return aggregated_explanations
