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
from examples.speech_to_text.occlusion_explanation.scorers.gender_scorer import GenderScorer


@register_scorer("gender_term_predicted_ratio")
class GenderTermPredictedRatioScorer(GenderScorer):
    """
    Assign the attribution scores based on the ratio between the original and
    perturbed probabilities of the gender term.
    Only gender terms are considered. If the gender term is composed of multiple tokens,
    the probabilities are aggregated with a product and a brevity penalty can be applied.
    """

    @staticmethod
    def get_prob_diff(
            gt_orig_probs: Tensor,
            gt_perturb_probs: Tensor) -> Tensor:
        """
        Args:
            - gt_orig_probs: original probabilities of the generated term.
            - gt_perturb_probs: perturbed probabilities of the generated term.
        Returns:
            Attribution scores based on the ratio between the original and
             perturbed probabilities of the gender term.
        """
        return (gt_orig_probs / gt_perturb_probs)
