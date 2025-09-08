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

from typing import Dict, Tuple
import logging

from torch import Tensor

from examples.speech_to_text.occlusion_explanation.scorers import register_scorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_scorer import GenderScorer


LOGGER = logging.getLogger(__name__)


@register_scorer("gender_term_predicted_diff")
class GenderTermPredictedDiffScorer(GenderScorer):
    """
    Assign the attribution scores based on the difference between the original and
    perturbed probabilities of the gender term of the constrained decoding.
    Only gender terms are considered. 
    """
    @staticmethod
    def get_prob_diff(
            gt_orig_probs: Tensor,
            gt_perturb_probs: Tensor) -> Tensor:
        """
        Args:
            - gt_orig_probs: Tensor of size (batch_size, 1, 1)
            - gt_perturb_probs: Tensor of size (batch_size, 1, 1)
        Returns:
            - Tensor of size (batch_size, 1, 1). There is only one score for each utterance
            in the batch, since only one gender term is annotated per utterance.
        """
        return gt_orig_probs - gt_perturb_probs

