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

from torch import Tensor

from examples.speech_to_text.occlusion_explanation.scorers import register_scorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_term_contrastive import GenderTermContrastive


@register_scorer("gender_term_contrastive_parity_ratio")
class GenderTermContrastiveParityRatioScorer(GenderTermContrastive):
    """
    Assign the attribution scores based on the formula in Jacovi et al. 2021
        (https://aclanthology.org/2021.emnlp-main.120.pdf).
    This scorer is referred to as the "contrastive relative scorer" in the paper "The Unheard Alternative:
        Contrastive Explanations for Speech-to-Text Models" (Conti et al., BlackboxNLP 2025).
    """
    @staticmethod
    def get_prob_diff(
            gt_orig_probs: Tensor,
            gt_perturb_probs: Tensor,
            swapped_gt_orig_probs: Tensor,
            swapped_gt_perturb_probs: Tensor) -> Tensor:
        """
        Compute the attribution scores based on the formula in Jacovi et al. 2021
        (https://aclanthology.org/2021.emnlp-main.120.pdf).
        Args:
            - gt_orig_probs: original probabilities of the generated term.
            - gt_perturb_probs: perturbed probabilities of the generated term.
            - swapped_gt_orig_probs: original probabilities of the swapped term.
            - swapped_gt_perturb_probs: perturbed probabilities of the swapped term.
        """
        return (gt_orig_probs / (gt_orig_probs + swapped_gt_orig_probs)) \
            - (gt_perturb_probs / (gt_perturb_probs + swapped_gt_perturb_probs))
