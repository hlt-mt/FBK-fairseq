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


@register_aggregator("time_aggregator")
class TimeLevelAggregator(Aggregator):
    """
    Aggregates explanations of filterbanks by averaging over time, thus preserving the
    channel dimension.

    Time spans can be specified for the aggregation, resulting in a  size transformation
    where the original explanation size of (seq_len, time, channels) becomes (seq_len,
    n_spans, channels). If no time spans are provided, the aggregation collapses the
    entire audio duration, resulting in a size of (seq_len, 1, channels).

    Explanations of target embedding are kept unchanged.
    """
    def __call__(
            self,
            explanations: Dict[int, Dict[str, Union[Tensor, str, List[str]]]],
            indices: Dict[int, List[Tuple[int, int]]] = None,
    ) -> Dict[int, Dict[str, Union[Tensor, List[str]]]]:
        """
        Performs aggregation. If `indices` are provided, aggregation is performed only
        along the specified timeframes. This can be useful when channels need to be
        aggregated only for specific speech sounds.

        The `indices` parameter should be a dictionary where each key corresponds to a
        sample ID, and the value is a tuple containing a list of tuples, where each tuple
        contains:
           - The token index to which the corresponding sound belongs.
           - A tuple indicating the start and end frames that compose the time span
             in which the corresponding sound is uttered.
        An example of `indices` for sample 1 is:
        {1: [(120, 125), (138, 146), (760, 761)], 2: [(145, 150)]}
        """
        if indices:
            for sample_id in list(explanations.keys()):
                sound_indices = indices.get(sample_id, None)
                if sound_indices:
                    fbank = explanations[sample_id]["fbank_heatmap"]
                    # Iterate over all time spans
                    aggregated_explanation = torch.cat([
                        fbank[:, frame_start:frame_end + 1].mean(dim=1, keepdim=True)
                        for frame_start, frame_end in sound_indices], dim=1)
                    explanations[sample_id]["fbank_heatmap"] = aggregated_explanation
                    explanations[sample_id]["tgt_embed_heatmap"] = \
                        explanations[sample_id]["tgt_embed_heatmap"]
                else:
                    del explanations[sample_id]

        else:  # indices not provided, aggregation performed over the entire audio duration
            for sample_id, explanation_data in explanations.items():
                aggregated_explanation = explanation_data["fbank_heatmap"].mean(dim=1, keepdim=True)
                explanations[sample_id]["fbank_heatmap"] = aggregated_explanation

        return explanations
