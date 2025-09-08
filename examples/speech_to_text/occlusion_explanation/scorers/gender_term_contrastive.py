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

from abc import abstractmethod
from typing import Dict, Tuple
import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.scorers.gender_scorer import GenderScorer


class GenderTermContrastive(GenderScorer):
    """
    Parent class for scorers that compare the probability of the predicted gender term 
    and that of its counterpart with swapped gender.
    """
    @staticmethod
    @abstractmethod
    def get_prob_diff(
            gt_orig_probs: Tensor,
            gt_perturb_probs: Tensor,
            swapped_gt_orig_probs: Tensor,
            swapped_gt_perturb_probs: Tensor) -> Tensor:
        """
        Compares the gender term probabilities obtained with the original input
        and those obtained with the perturbed input for the predicted and swapped gender terms.
        """
        pass
    
    def __call__(
            self,
            sample: Dict,
            orig_probs: Dict,
            perturb_probs: torch.Tensor,
            tgt_embed_masks: torch.Tensor,
            swapped_perturb_probs: torch.Tensor,
            *args,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The call method is overriden for the contrastive gender scorers so that it computes the word 
        # probabilities for both the gender term and the swapped term.
        
        # Change the original probabilities to the format expected by get_padded_probs
        swapped_orig_probs = {
            int(k[:-8]): v for k, v in orig_probs.items() if (type(k) == str and k.endswith("_swapped"))}
        orig_probs = {k: v for k, v in orig_probs.items() if not (type(k) == str and k.endswith("_swapped"))}

        # Get the padded probabilities
        padded_orig_probs, padded_perturbed_probs = self.get_padded_probs(
            orig_probs, perturb_probs, sample["orig_id"], sample["target_lengths"])
        padded_swapped_orig_probs, padded_swapped_perturbed_probs = self.get_padded_probs(
            swapped_orig_probs,
            swapped_perturb_probs,
            sample["orig_id"],
            sample["swapped_target_lengths"])
        
        # Aggregate the probabilities at word level
        gt_orig_probs = self.prob_aggregator.compute_word_probability(
            padded_orig_probs, sample["target"], sample["gender_term_starts"], sample["gender_term_ends"])
        gt_perturb_probs = self.prob_aggregator.compute_word_probability(
            padded_perturbed_probs, sample["target"], sample["gender_term_starts"], sample["gender_term_ends"])
        swapped_gt_orig_probs = self.prob_aggregator.compute_word_probability(
            padded_swapped_orig_probs, sample["swapped_target"], sample["gender_term_starts"], sample["swapped_term_ends"])
        swapped_gt_perturb_probs = self.prob_aggregator.compute_word_probability(
            padded_swapped_perturbed_probs, sample["swapped_target"], sample["gender_term_starts"], sample["swapped_term_ends"])
        
        # Compute the scores and turn them into heatmaps
        scores = self.get_prob_diff(
            gt_orig_probs,
            gt_perturb_probs,
            swapped_gt_orig_probs,
            swapped_gt_perturb_probs)

        single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks = \
            self.get_heatmaps(scores, sample["masks"], tgt_embed_masks)
        single_tgt_embed_heatmaps = self._make_heatmaps_causal(single_tgt_embed_heatmaps, sample)
        return single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks
