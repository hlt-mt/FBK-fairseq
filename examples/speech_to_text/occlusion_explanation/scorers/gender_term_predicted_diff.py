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
import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.scorers import register_scorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_scorer import GenderScorer


@register_scorer("gender_term_predicted_diff")
class GenderTermPredictedDiffScorer(GenderScorer):
    """
    Assign the attribution scores based on the difference between the original and
    perturbed probabilities of the gender term of the constrained decoding.
    Only gender terms are considered. If the gender term is composed of multiple tokens,
    the probabilities are aggregated with a product and a brevity penalty is applied.
    """
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
        # Select the probability of the generated tokens
        tgt_tokens = sample["target"]
        token_orig_probs = torch.take_along_dim(orig_probs, tgt_tokens.unsqueeze(-1), dim=2) # (B, S, 1)
        token_perturb_probs = torch.take_along_dim(perturb_probs, tgt_tokens.unsqueeze(-1), dim=2) # (B, S, 1)
        
        # Select the probability of the gender terms
        sample_size = len(sample["gender_terms_indices"])
        assert sample_size == token_orig_probs.size(0)
        gender_term_length, gt_orig_probs, gt_perturb_probs = [], [], []
        for i in range(sample_size):
            start, end = sample["gender_terms_indices"][i].split("-")
            start, end = int(start), int(end)
            gender_term_length.append(end - start + 1)
            gt_orig_probs.append(token_orig_probs[i, start : end + 1].prod())
            gt_perturb_probs.append(token_perturb_probs[i, start : end + 1].prod())

        # Convert the lists to tensors
        gender_term_length = torch.tensor(gender_term_length, device=orig_probs.device).view(-1, 1, 1)
        gt_orig_probs = torch.stack(gt_orig_probs).view(-1, 1, 1)
        gt_perturb_probs = torch.stack(gt_perturb_probs).view(-1, 1, 1)

        # We take the nth root (where n is the number of tokens in the term) of the product of the 
        # token probabilities as a form of brevity penalty to avoid penalizing longer terms
        gt_orig_probs = gt_orig_probs ** (1 / gender_term_length)
        gt_perturb_probs = gt_perturb_probs ** (1 / gender_term_length)

        return gt_orig_probs - gt_perturb_probs
