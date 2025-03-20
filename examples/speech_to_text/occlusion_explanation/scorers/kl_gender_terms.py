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
from torch import Tensor
import torch

from examples.speech_to_text.occlusion_explanation.scorers import register_scorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_scorer import GenderScorer


@register_scorer("KL_gender_terms")
class KLGenderScorer(GenderScorer):
    """
    For the annotated gender term in each utterance, compute the difference between the averaged
    original and perturbed probability distributions, using KL divergence.
    """

    @staticmethod
    def kl_divergence(
            orig_probs: Tensor,
            perturb_probs: Tensor) -> Tensor:
        """
        Compute KL divergence between the two probability distributions.
        """
        orig_entropy = orig_probs * orig_probs.log()
        cross_entropy = orig_probs * perturb_probs.log()
        return (orig_entropy - cross_entropy).sum(dim=2).unsqueeze(-1)

    @staticmethod
    def get_prob_diff(
            orig_probs: Tensor,
            perturb_probs: Tensor,
            sample: Dict) -> Tensor:
        """
        Args:
            - orig_probs: Tensor of size (batch_size, seq_len, vocab_size)
            - perturb_probs: Tensor of size (batch_size, seq_len, vocab_size)
            - sample: dictionary containing a field 'gender_terms_indices' 
                      with a list of strings containing the start and end indices of 
                      the gender term in each utterance.
        Returns:
            - Tensor of size (batch_size, 1, 1). There is only one score for each utterance
              in the batch, since only one gender term is annotated per utterance.
        """
        batch_size, seq_len, vocab_size = orig_probs.size()

        # Compute the KL divergence
        token_scores = KLGenderScorer.kl_divergence(orig_probs, perturb_probs)
        assert token_scores.size() == (batch_size, seq_len, 1)

        # Aggregate scores for each gender term using max
        # For multi-token gender terms, compute KL divergence for each token and use the highest value
        # (even if only one token is highly impacted by the perturbation, the score is high)
        gt_scores = torch.zeros(batch_size, device=orig_probs.device)
        for batch, indices_range in enumerate(sample['gender_terms_indices']):
            start, end = indices_range.split('-')
            start, end = int(start), int(end)
            gt_scores[batch] = token_scores[batch,start:end+1].max()

        return gt_scores.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
