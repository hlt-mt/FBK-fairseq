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

import importlib
import os
import logging
from abc import abstractmethod
from typing import Tuple, Dict

import torch
from torch import Tensor


LOGGER = logging.getLogger(__name__)

SCORER_REGISTRY = {}
SCORER_CLASS_NAMES = set()


class Scorer:
    """
    Assign the attribution scores based on the difference between the original and
    perturbed probabilities.
    """
    @staticmethod
    def get_padded_probs(
            orig_probs: Dict,
            perturb_probs: Tensor,
            orig_indices: Tensor,
            target_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pad the original probabilities to the max sequence lengths of the perturbed probabilities,
        since the original probabilities stored in orig_probs have always a sequence length lesser
        or equal to that of the probabilities in perturb_probs. It also set probabilities of <pad>
        set to 0 in the perturbed probabilities.

        Args:
            - orig_probs: dictionary in the form {"id": Tensor (padded_seq_len, dict_len)}
            - perturb_probs: Tensor of size (batch_size, padded_seq_len, dict_len)
            - orig_indices: Tensor of size (batch_size)
            - target_lengths: Tensor of size (batch_size)
        """
        batch_size, max_length, vocab_size = perturb_probs.size()
        seq_range = torch.arange(max_length, device=perturb_probs.device)
        mask = seq_range.unsqueeze(0) >= target_lengths.unsqueeze(1)
        perturb_probs[mask] = 0  # set padded regions to 0 in batched_perturb_probs

        unique_indices, counts = torch.unique_consecutive(orig_indices, return_counts=True)
        batched_orig_probs = torch.zeros(
            (batch_size, max_length, vocab_size), dtype=torch.float, device=perturb_probs.device)
        prev_count = 0
        for orig_ind, count in zip(unique_indices, counts):
            single_orig_prob = orig_probs[str(orig_ind.item())]
            if single_orig_prob.numel() > 0:  # check if not empty
                # assign the original probabilities to the corresponding
                # batch items in the empty padded matrix
                batched_orig_probs[prev_count:prev_count + count, : single_orig_prob.size(0), :] = \
                    single_orig_prob
            prev_count += count

        return batched_orig_probs, perturb_probs

    @staticmethod
    def get_heatmaps(
            scores: Tensor,
            fbank_masks: Tensor,
            tgt_embed_masks: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return 4D heatmaps and masks for both filterbanks and target embeddings.
        Heatmaps are obtained through element-wise multiplication between scores and masks.
        These tensors undergo (un)squeeze operations which make them broadcastable.

        The shapes of the masks for the filterbank and the target embeddings can have various shapes.
        If they are derived from continuous or slic-based perturbations they are:
        - fbank_masks: (batch size, time, channels)
        - tgt_embed_masks: (batch size, sequence length, embedding dimension)
        If they are derived from discrete perturbations they are:
        - fbank_masks: (batch size, time)
        - tgt_embed_masks: (batch size, sequence length)
        To account for this variability, an initial condition with an unsqueeze operation is made;
        thus, the heatmaps are always 4D.

        The tensor containing the scores to be assigned to the masks, instead, has shape
        (batch size, sequence length, 1).
        """
        # Expand heatmaps with dimensions (batch_size, time/seq_length) to have an
        # additional dimension (batch_size, time/seq_length, 1)
        if len(fbank_masks.size()) < 3:
            fbank_masks = fbank_masks.unsqueeze(-1)
        if len(tgt_embed_masks.size()) < 3:
            tgt_embed_masks = tgt_embed_masks.unsqueeze(-1)

        # Expand heatmaps with dimensions (batch_size, time/seq_length, 1) or
        # (batch_size, time/sequence_length, channels/embed_dim/1) to have an
        # additional dimension (batch_size, 1, time/sequence_length, channels/embed_dim/1)
        fbank_masks = fbank_masks.unsqueeze(1)
        tgt_embed_masks = tgt_embed_masks.unsqueeze(1)

        # logical negation as scores as assigned to zeroed out elements
        fbank_masks = 1 - fbank_masks
        tgt_embed_masks = 1 - tgt_embed_masks

        # Expand scores with dimensions (batch size, sequence length, 1) to have an
        # additional dimension (batch size, sequence length, 1, 1)
        scores = scores.unsqueeze(-1)

        single_fbank_heatmaps = fbank_masks * scores
        single_tgt_embed_heatmaps = tgt_embed_masks * scores

        return single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks

    @staticmethod
    def _make_heatmaps_causal(heatmaps: Tensor) -> Tensor:
        """
        Enforces causality in the tgt_embed_heatmaps by zeroing out the embeddings related
        to the tokens after the one that is being considered. This is needed because the
        masks are created for the entire sequence to be generated and used also for the
        padded tokens. Heatmaps have shape (batch, seq_len, seq_len, embed_dim/1), where
        seq_len is the length of the longest sequence (the other sequences are padded).
        """
        sequence_length = heatmaps.shape[1]
        mask = torch.tril(torch.ones((sequence_length, sequence_length), device=heatmaps.device))
        heatmaps *= mask.unsqueeze(-1).unsqueeze(0)
        return heatmaps

    @staticmethod
    @abstractmethod
    def get_prob_diff(*args, **kwargs) -> Tensor:
        """
        Computes the difference between the probabilities obtained with the original input
        and those obtained with the perturbed input.
        """
        pass

    def __call__(
            self,
            sample: Dict,
            orig_probs: Dict,
            perturb_probs: Tensor,
            tgt_embed_masks: Tensor,
            *args,
            **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        padded_orig_probs, padded_perturbed_probs = self.get_padded_probs(
            orig_probs, perturb_probs, sample["orig_id"], sample["net_input"]["target_lengths"])
        scores = self.get_prob_diff(padded_orig_probs, padded_perturbed_probs, sample["net_input"]["target"])
        single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks = \
            self.get_heatmaps(scores, sample["masks"], tgt_embed_masks)
        single_tgt_embed_heatmaps = self._make_heatmaps_causal(single_tgt_embed_heatmaps)
        return single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks


def register_scorer(name):
    def register_scorer_cls(cls):
        if name in SCORER_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate scorer ({name})")
        if not issubclass(cls, Scorer):
            raise ValueError(
                f"Scorer ({name}: {cls.__name__}) must extend Scorer")
        if cls.__name__ in SCORER_CLASS_NAMES:
            raise ValueError(
                f"Cannot register scorer with duplicate class name ({cls.__name__})")
        SCORER_REGISTRY[name] = cls
        SCORER_CLASS_NAMES.add(cls.__name__)
        LOGGER.info(f"Scorer used: {name}.")
        return cls
    return register_scorer_cls


def get_scorer(name):
    return SCORER_REGISTRY[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module(
            'examples.speech_to_text.occlusion_explanation.scorers.' + module_name)
