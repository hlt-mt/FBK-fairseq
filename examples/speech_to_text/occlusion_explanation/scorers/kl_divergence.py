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

from torch import Tensor

from examples.speech_to_text.occlusion_explanation.scorers import register_scorer, Scorer


@register_scorer("KL")
class KLScorer(Scorer):
    """
    Compute the difference between the original and perturbed probability distributions,
    using KL divergence.
    """
    @staticmethod
    def get_prob_diff(
            orig_probs: Tensor,
            perturb_probs: Tensor,
            *args) -> Tensor:
        """
        Compute KL divergence between the two probabiity distributions.
        """
        orig_entropy = orig_probs * orig_probs.log()
        cross_entropy = orig_probs * perturb_probs.log()
        return (orig_entropy - cross_entropy).sum(dim=2).unsqueeze(-1)
