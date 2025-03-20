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

from typing import Dict

import torch
from examples.speech_to_text.occlusion_explanation.scorers import Scorer


class GenderScorer(Scorer):
    """
    Parent class for different scorers that focus on explaining only the gender term in each utterance.
    """

    @staticmethod
    def _make_heatmaps_causal(heatmaps: torch.Tensor, sample: Dict) -> torch.Tensor:
        """
        Enforces causality in the tgt_embed_heatmaps by zeroing out the embeddings related
        to the tokens after the one that is being considered. This is needed because the
        masks are created for the entire sequence to be generated and used also for the
        padded tokens. Heatmaps have shape (batch, seq_len, seq_len, embed_dim/1), where
        seq_len is the length of the longest sequence (the other sequences are padded).
        Args:
            - heatmaps: Tensor of size (batch_size, 1, seq_len, embed_dim/1)
                        (The second dimension is 1 because we are explaining a single term per utterance).
            - sample: dictionary containing a field 'gender_terms_indices' 
                      with a list of strings containing the start and end indices of 
                      the gender term in each utterance.
        Returns:
            - Tensor of size (batch_size, 1, seq_len, embed_dim/1) corresponding to
              the causal heatmaps (explanations for tokens including and following the gender
              term being studied are masked).
        """
        mask = torch.zeros(heatmaps.shape, device=heatmaps.device) # (batch_size, 1, seq_len, embed_dim/1)
        for batch, indices_range in enumerate(sample['gender_terms_indices']):
            start, end = indices_range.split('-')
            start, end = int(start), int(end)
            mask[batch][0][0:start + 1, :] = 1. # start + 1 because of the bos token 

        heatmaps *= mask
        return heatmaps
