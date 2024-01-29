# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch
from torch import Tensor


LOGGER = logging.getLogger(__name__)


class OcclusionFbankPerturbator(ABC):
    """
    Base class to perform occlusion perturbations of filterbanks.
    """
    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: Dict = None):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        pass


class OcclusionDecoderEmbeddingsPerturbator(ABC):
    """
    Base class to perform occlusion perturbations of target embeddings.
    """
    def __init__(
            self, no_position_occlusion: bool = False, p: float = 0.5):
        self.no_position_occlusion = no_position_occlusion
        self.p = p
        LOGGER.info(f"Masking decoder embeddings with probability of {self.p}.")

    @classmethod
    def from_config_dict(cls, config: Dict = None):
        if config is None:
            return cls(no_position_occlusion=True, p=0.5)
        else:
            _config = config
            decoder_occlusion = _config.get("decoder_occlusion", {})
            no_position_occlusion = decoder_occlusion.get("no_position_occlusion", True)
            p = decoder_occlusion.get("p", 0.5)
            assert 0. < p <= 1.
            return cls(no_position_occlusion, p)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        pass


PERTURBATION_REGISTRY = {}
PERTURBATION_CLASS_NAMES = set()


def register_perturbator(name):
    def register_perturbation_cls(cls):
        if name in PERTURBATION_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate perturbation ({name})")
        if not issubclass(cls, (OcclusionFbankPerturbator, OcclusionDecoderEmbeddingsPerturbator)):
            raise ValueError(
                f"Perturbation ({name}: {cls.__name__}) must extend "
                "OcclusionFbankPerturbator or OcclusionDecoderEmbeddingsPerturbator")
        if cls.__name__ in PERTURBATION_CLASS_NAMES:
            raise ValueError(
                f"Cannot register perturbation with duplicate class name ({cls.__name__})")
        PERTURBATION_REGISTRY[name] = cls
        PERTURBATION_CLASS_NAMES.add(cls.__name__)
        LOGGER.info(f"Occlusion perturbation applied: {name}.")
        return cls
    return register_perturbation_cls


def get_perturbator(name):
    return PERTURBATION_REGISTRY[name]


class Scorer:
    """
    Assign the attribution scores based on the difference between the original and
    perturbed probabilities.
    """
    @staticmethod
    def get_padded_probs(
            orig_probs: Dict,  # {"id": Tensor (padded_seq_len, dict_len)}
            perturb_probs: Tensor,  # (batch_size, padded_seq_len, dict_len)
            orig_indices: Tensor,
            target_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pad the original probabilities to the max sequence lengths of the perturbed probabilities,
        since the original probabilities stored in orig_probs have always a sequence length lesser or equal
        to that of the probabilities in perturb_probs. It also set probabilities of <pad> set to 0 in
        the perturbed probabilities.
        """
        batch_size, max_length, vocab_size = perturb_probs.size()
        batched_orig_probs = torch.zeros(
            (batch_size, max_length, vocab_size), dtype=torch.long, device=perturb_probs.device)
        batched_perturb_probs = batched_orig_probs
        for i in range(batch_size):
            single_orig_prob = orig_probs[str(orig_indices[i].item())]
            single_perturb_prob = perturb_probs[i, :target_lengths[i].item(), :]
            if single_orig_prob.numel() > 0 and single_perturb_prob.numel():  # Check if not empty
                batched_orig_probs[i, :single_orig_prob.size(0), :] = single_orig_prob
                batched_perturb_probs[i, :single_perturb_prob.size(0), :] = single_perturb_prob
        return batched_orig_probs, batched_perturb_probs

    @staticmethod
    def get_heatmaps(
            scores: Tensor,
            fbank_masks: Tensor,
            tgt_embed_masks: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Obtains heatmaps through element-wise multiplication between prob difference and masks.

        The shapes of the masks and probabilities Tensors are:
            - fbank_masks: (batch size, time, channels)
            - tgt_embed_masks: (batch size, sequence length, embedding dimension)
            - probs: (batch size, sequence length, vocabulary size).
        From probs, toke_probs with shape (batch size, sequence length, 1) is obtained.
        In order to make these Tensors broadcastable and to obtain outputs with shape
        (batch size, sequence length, time/sequence length, channels/embedding dimension),
        namely single 2D heatmaps for each batch and for each token, their shapes become
            - fbank_masks: (batch size, 1, time, channels)
            - tgt_embed_masks: (batch size, 1, sequence length, embedding dimension)
            - token_probs: (batch size, sequence length, 1, 1).
        """
        # Making masks with shape (batch_size, 1, time/seq_length, channels/embedding_dim)
        fbank_masks = 1 - fbank_masks
        tgt_embed_masks = 1 - tgt_embed_masks
        fbank_masks = fbank_masks.unsqueeze(1)
        tgt_embed_masks = tgt_embed_masks.unsqueeze(1)
        single_fbank_heatmaps = fbank_masks * scores.unsqueeze(-1)
        single_tgt_embed_heatmaps = tgt_embed_masks * scores.unsqueeze(-1)
        return single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks

    @staticmethod
    def _make_heatmaps_causal(heatmaps: Tensor) -> Tensor:
        """
        Enforces causality in the tgt_embed_heatmaps by zeroing out the embeddings related
        to the tokens after the one that is being considered. This is needed because the
        masks are created for the entire sequence to be generated and used also for the
        padded tokens; they have shape (batch, seq_len, seq_len, embed_dim), where seq_len
        is the length of the longest sequence (the other sequences are padded).
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


SCORER_REGISTRY = {}
SCORER_CLASS_NAMES = set()


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
