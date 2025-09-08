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
import logging
from typing import Dict, Tuple

import torch
from examples.speech_to_text.occlusion_explanation.scorers import Scorer
from examples.speech_to_text.occlusion_explanation.scorers.probability_aggregators import (
    PROB_AGGREGATION_REGISTRY, ProbabilityAggregator, get_prob_aggregator)


LOGGER = logging.getLogger(__name__)


class GenderScorer(Scorer):
    """
    Parent class for different scorers that focus on explaining only the gender term in each utterance.
    """
    def __init__(self, prob_aggregator: ProbabilityAggregator) -> None:
        """
        Args:
            - prob_aggregator: The probability aggregator to be used to compute 
                the probability of the gender term under study.
        """
        self.prob_aggregator = prob_aggregator

    @classmethod
    def from_config_dict(cls, config: Dict = None, tgt_dict: Dict = None):
        aggregator_category = config.get("scorer", {}).get("prob_aggregation", "word_boundary")
        assert aggregator_category in PROB_AGGREGATION_REGISTRY, \
            f"Invalid probability aggregator category: {aggregator_category}. " \
            f"Available categories: {PROB_AGGREGATION_REGISTRY}."
        prob_aggregator = get_prob_aggregator(aggregator_category)(tgt_dict)
        LOGGER.info(f"Probability aggregator used: {aggregator_category}.")
        return cls(prob_aggregator)

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
            - sample: dictionary containing a field 'gender_term_starts' 
                      with a list of the start indices of the gender term in each utterance.
        Returns:
            - Tensor of size (batch_size, 1, seq_len, embed_dim/1) corresponding to
              the causal heatmaps (explanations for tokens including and following the gender
              term being studied are masked).
        """
        mask = torch.zeros(heatmaps.shape, device=heatmaps.device) # (batch_size, 1, seq_len, embed_dim/1)
        for batch, start in enumerate(sample['gender_term_starts']):
            mask[batch][0][0:start + 1, :] = 1. # start + 1 because of the bos token 

        heatmaps *= mask
        return heatmaps
    
    @staticmethod
    @abstractmethod
    def get_prob_diff(
        gt_orig_probs: torch.Tensor, gt_perturb_probs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes the difference between the gender term probabilities obtained with the original input
        and those obtained with the perturbed input.
        """
        pass

    def __call__(
            self,
            sample: Dict,
            orig_probs: Dict,
            perturb_probs: torch.Tensor,
            tgt_embed_masks: torch.Tensor,
            *args,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The call method is overriden for the gender scorers so that it computes the word 
        # probabilities for the gender term.
            
        # Get the padded probabilities for the predicted terms
        padded_orig_probs, padded_perturbed_probs = self.get_padded_probs(
            orig_probs, perturb_probs, sample["orig_id"], sample["target_lengths"])
        
        # Aggregate predicted term probabilities at word level
        gt_orig_probs = self.prob_aggregator.compute_word_probability(
            padded_orig_probs, sample["target"], sample["gender_term_starts"], sample["gender_term_ends"])
        gt_perturb_probs = self.prob_aggregator.compute_word_probability(
            padded_perturbed_probs, sample["target"], sample["gender_term_starts"], sample["gender_term_ends"])
        
        # Compute the scores and turn them into heatmaps
        scores = self.get_prob_diff(gt_orig_probs, gt_perturb_probs)

        single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks = \
            self.get_heatmaps(scores, sample["masks"], tgt_embed_masks)
        single_tgt_embed_heatmaps = self._make_heatmaps_causal(single_tgt_embed_heatmaps, sample)
        return single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks
