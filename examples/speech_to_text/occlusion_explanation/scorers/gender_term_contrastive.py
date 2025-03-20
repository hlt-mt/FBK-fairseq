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

from typing import Dict, Tuple
import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.scorers import register_scorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_scorer import GenderScorer


@register_scorer("gender_term_contrastive")
class GenderTermContrastiveScorer(GenderScorer):
    """
    Assign the attribution scores based on the difference between the 
    difference between the original probability and the perturbed probability of the generated term
    and the difference between the original probability and the perturbed probability of the swapped term.
    Only gender terms are considered.
    """
    @staticmethod
    def get_prob_diff(
            orig_probs: Tensor,
            perturb_probs: Tensor,
            swapped_orig_probs: Tensor,
            swapped_perturb_probs: Tensor,
            sample: Dict) -> Tensor:
        """
        Args:
            - orig_probs: Tensor of size (batch_size, seq_len, vocab_size)
            - perturb_probs: Tensor of size (batch_size, seq_len, vocab_size)
            - swapped_orig_probs: Tensor of size (batch_size, seq_len, vocab_size)
            - swapped_perturb_probs: Tensor of size (batch_size, seq_len, vocab_size)
            - sample: dictionary containing a field 'gender_terms_indices' 
                      with a list of strings containing the start and end indices of 
                      the gender term in each utterance.
        Returns:
            - Tensor of size (batch_size, 1, 1). There is only one score for each utterance
            in the batch, since only one gender term is annotated per utterance.
        """
        # Select the probability of the generated tokens
        tgt_tokens = sample["target"]
        token_orig_probs = torch.take_along_dim(orig_probs, tgt_tokens.unsqueeze(-1), dim=2)  # (B, S, 1)
        token_perturb_probs = torch.take_along_dim(perturb_probs, tgt_tokens.unsqueeze(-1), dim=2) # (B, S, 1)

        swapped_tgt_tokens = sample["swapped_target"]
        swapped_token_orig_probs = torch.take_along_dim(
            swapped_orig_probs, swapped_tgt_tokens.unsqueeze(-1), dim=2) # (B, S, 1)
        swapped_token_perturb_probs = torch.take_along_dim(
            swapped_perturb_probs, swapped_tgt_tokens.unsqueeze(-1), dim=2) # (B, S, 1)
        
        # Select the probability of the gender terms
        sample_size = len(sample["gender_terms_indices"])
        assert sample_size == token_orig_probs.size(0)
        mask_gender_terms = torch.zeros_like(token_orig_probs, dtype=torch.bool)
        mask_swapped_terms = torch.zeros_like(swapped_token_orig_probs, dtype=torch.bool)
        gender_term_length, swapped_gender_term_length = [], []
        gt_orig_probs, gt_perturb_probs, swapped_gt_orig_probs, swapped_gt_perturb_probs = [], [], [], []
        for i in range(sample_size):
            # This code assumes that there is only one gender term per utterance
            start, end = sample["gender_terms_indices"][i].split("-")
            start, end = int(start), int(end)
            gt_orig_probs.append(token_orig_probs[i, start:end + 1].prod())
            gt_perturb_probs.append(token_perturb_probs[i, start:end + 1].prod())
            gender_term_length.append(end - start + 1)
            end_swapped = end + (sample["swapped_target_lengths"][i] - sample["target_lengths"][i])
            swapped_gt_orig_probs.append(swapped_token_orig_probs[i, start:end_swapped + 1].prod())
            swapped_gt_perturb_probs.append(swapped_token_perturb_probs[i, start:end_swapped + 1].prod())
            swapped_gender_term_length.append(end_swapped - start + 1)

        # Convert the lists to tensors
        gender_term_length = torch.tensor(gender_term_length, device=token_orig_probs.device).view(-1, 1, 1)
        swapped_gender_term_length = torch.tensor(swapped_gender_term_length, device=token_orig_probs.device).view(-1, 1, 1)

        gt_orig_probs = torch.stack(gt_orig_probs).view(-1, 1, 1)
        gt_perturb_probs = torch.stack(gt_perturb_probs).view(-1, 1, 1)
        swapped_gt_orig_probs = torch.stack(swapped_gt_orig_probs).view(-1, 1, 1)
        swapped_gt_perturb_probs = torch.stack(swapped_gt_perturb_probs).view(-1, 1, 1)

        # We take the nth root (where n is the number of tokens in the term) of the product of the 
        # token probabilities as a form of brevity penalty to avoid penalizing longer terms
        gt_orig_probs = gt_orig_probs ** (1 / gender_term_length)
        gt_perturb_probs = gt_perturb_probs ** (1 / gender_term_length)
        swapped_gt_orig_probs = swapped_gt_orig_probs ** (1 / swapped_gender_term_length)
        swapped_gt_perturb_probs = swapped_gt_perturb_probs ** (1 / swapped_gender_term_length)

        return (gt_orig_probs - gt_perturb_probs) - (swapped_gt_orig_probs - swapped_gt_perturb_probs)
    
    def __call__(
            self,
            sample: Dict,
            orig_probs: Dict,
            perturb_probs: Tensor,
            tgt_embed_masks: Tensor,
            swapped_perturb_probs: Tensor,
            *args,
            **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Change the original probabilities to the format expected by get_padded_probs
        swapped_orig_probs = {
            int(k[:-8]): v for k, v in orig_probs.items() if (type(k) == str and k.endswith("_swapped"))}
        orig_probs = {k: v for k, v in orig_probs.items() if not (type(k) == str and k.endswith("_swapped"))}

        # Get the padded probabilities
        padded_orig_probs, padded_perturbed_probs = self.get_padded_probs(
            orig_probs, perturb_probs, sample["orig_id"],
            sample["target_lengths"])
        padded_swapped_orig_probs, padded_swapped_perturbed_probs = self.get_padded_probs(
            swapped_orig_probs,
            swapped_perturb_probs,
            sample["orig_id"],
            sample["swapped_target_lengths"])
        
        # Compute the scores and turn them into heatmaps
        scores = self.get_prob_diff(
            padded_orig_probs,
            padded_perturbed_probs,
            padded_swapped_orig_probs,
            padded_swapped_perturbed_probs,
            sample)
        single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks = \
            self.get_heatmaps(scores, sample["masks"], tgt_embed_masks)
        single_tgt_embed_heatmaps = self._make_heatmaps_causal(single_tgt_embed_heatmaps, sample)
        return single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks
