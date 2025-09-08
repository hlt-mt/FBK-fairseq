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

from abc import ABC, abstractmethod
import logging
import string
from typing import Dict, List
import torch


LOGGER = logging.getLogger(__name__)

PROB_AGGREGATION_REGISTRY = {}
PROB_AGGREGATION_NAMES = set()


class ProbabilityAggregator(ABC):
    """
    Interface for probability aggregators that compute the probability of a word 
    from the probability distribution output by the model over a vocabulary of subwords.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the probability aggregator. Can be overridden as needed by subclasses.
        """
        super().__init__()

    @abstractmethod
    def compute_word_probability(
            self,
            probs: torch.Tensor,
            tgt_tokens: torch.LongTensor,
            start_indices: torch.Tensor,
            *args) -> torch.Tensor:
        """
        Abstract method to be implemented based on the aggregation strategy to compute
        the probability of a word from the probability of its subword tokens.
        Args:
            - probs:    The tensor containing the probability distributions over the vocabulary
                output by the model (B x S x V).
            - tgt_tokens:   The tensor containing the tokens predicted (B x S).
            - start_indices:    List of length B containing the indices of the start of the word of interest.
        Returns:
            - The probability of the word of interest for each item in the batch (B x 1 x 1).
        """
        pass


def register_prob_aggregator(name):
    def register_prob_aggregator_cls(cls):
        if name in PROB_AGGREGATION_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate probability aggregator ({name})")
        if not issubclass(cls, ProbabilityAggregator):
            raise ValueError(
                f"ProbabilityAggregator ({name}: {cls.__name__}) must extend ProbabilityAggregator")
        if cls.__name__ in PROB_AGGREGATION_NAMES:
            raise ValueError(
                f"Cannot register probability aggregator with duplicate class name ({cls.__name__})")
        PROB_AGGREGATION_REGISTRY[name] = cls
        PROB_AGGREGATION_NAMES.add(cls.__name__)
        LOGGER.debug(f"Probability aggregator registered: {name}.")
        return cls
    return register_prob_aggregator_cls


def get_prob_aggregator(name):
    return PROB_AGGREGATION_REGISTRY[name]


@register_prob_aggregator("chain_rule")
class ChainProbabilityAggregator(ProbabilityAggregator):
    """
    Aggregator that computes the probability of a word by multiplying the probabilities
    of its subword tokens.
    """
    def compute_word_probability(
            self,
            probs: torch.Tensor,
            tgt_tokens: torch.LongTensor,
            start_indices: torch.Tensor,
            end_indices: torch.Tensor) -> torch.Tensor:
        """
        Computes the word probability from the subword probabilities by applying a simple
        chain rule, i.e., multiplying the probabilities of the subwords that compose a word.
        Args:
            - probs:    The tensor containing the probability distributions over the vocabulary
                output by the model (B x S x V).
            - tgt_tokens:   The tensor containing the tokens predicted (B x S).
            - start_indices:    Tensor of length B containing the indices of the start of the word of interest.
            - end_indices:  Tensor of length B containing the indices of the end of the word of interest.
        Returns:
            - The probability of the word of interest for each item in the batch (B x 1 x 1).
        """
        sample_size = len(start_indices)
        assert sample_size == probs.size(0)
        seq_len = probs.size(1)
        assert 0 <= torch.all(start_indices) and torch.all(start_indices < seq_len), \
            "Start indices must be within the sequence length."
        assert 0 <= torch.all(end_indices) and torch.all(end_indices < seq_len), \
            "End indices must be within the sequence length."
        token_probs = torch.take_along_dim(probs, tgt_tokens.unsqueeze(-1), dim=2)  # (batch_size, seq_len, 1)
        word_probs = []
        for i in range(sample_size):
            word_probs.append(token_probs[i, start_indices[i].item():end_indices[i].item() + 1].prod())
        return torch.stack(word_probs).view(-1, 1, 1)
    

@register_prob_aggregator("length_norm")
class LengthNormProbabilityAggregator(ProbabilityAggregator):
    """
    Aggregator that computes the probability of a word by taking the nth root 
    (where n is the number of subword tokens in the word) of the product of the 
    token probabilities as a form of brevity penalty to avoid penalizing longer terms,
    similar to the length normalization typically used in beam search.
    """
    def compute_word_probability(
            self,
            probs: torch.Tensor,
            tgt_tokens: torch.LongTensor,
            start_indices: torch.Tensor,
            end_indices: torch.Tensor) -> torch.Tensor:
        """
        Computes the word probability from the subword probabilities by applying a length
        normalization, i.e., taking the nth root of the product of the probabilities of the
        subwords that compose a word, where n is the number of subwords in the word.
        Args:
            - probs:    The tensor containing the probability distributions over the vocabulary
                output by the model (B x S x V).
            - tgt_tokens:   The tensor containing the tokens predicted (B x S).
            - start_indices:    Tensor of length B containing the indices of the start of the word of interest.
            - end_indices:  Tensor of length B containing the indices of the end of the word of interest.
        Returns:
            - The probability of the word of interest for each item in the batch (B x 1 x 1).
        """
        sample_size = len(start_indices)
        assert sample_size == probs.size(0)
        seq_len = probs.size(1)
        assert 0 <= torch.all(start_indices) and torch.all(start_indices < seq_len), \
            "Start indices must be within the sequence length."
        assert 0 <= torch.all(end_indices) and torch.all(end_indices < seq_len), \
            "End indices must be within the sequence length."
        token_probs = torch.take_along_dim(probs, tgt_tokens.unsqueeze(-1), dim=2)  # (batch_size, seq_len, 1)
        word_probs = []
        word_lengths = []
        for i in range(sample_size):
            word_probs.append(token_probs[i, start_indices[i].item():end_indices[i].item() + 1].prod())
            word_lengths.append(end_indices[i] - start_indices[i] + 1)
        word_probs = torch.stack(word_probs).view(-1, 1, 1)
        word_lengths = torch.stack(word_lengths).view(-1, 1, 1)
        return word_probs ** (1 / word_lengths)


@register_prob_aggregator("first_only")
class FirstOnlyProbabilityAggregator(ProbabilityAggregator):
    """
    Aggregator that uses the probability of the first subword token of a word
    as a proxy for that word's probability.
    """
    def compute_word_probability(
            self,
            probs: torch.Tensor,
            tgt_tokens: torch.LongTensor,
            start_indices: torch.Tensor,
            *args) -> torch.Tensor:
        """
        Computes the word probability from the subword probabilities by taking the probability
        of the first subword token of the word.
        Args:
            - probs:    The tensor containing the probability distributions over the vocabulary
                output by the model (B x S x V).
            - tgt_tokens:   The tensor containing the tokens predicted (B x S).
            - start_indices:    Tensor of length B containing the indices of the start of the word of interest.
        Returns:
            - The probability of the word of interest for each item in the batch (B x 1 x 1).
        """
        sample_size = len(start_indices)
        assert sample_size == probs.size(0)
        seq_len = probs.size(1)
        assert 0 <= torch.all(start_indices) and torch.all(start_indices < seq_len), \
            "Start indices must be within the sequence length."
        token_probs = torch.take_along_dim(probs, tgt_tokens.unsqueeze(-1), dim=2)  # (batch_size, seq_len, 1)
        start_indices = start_indices.view(-1, 1, 1)
        word_probs = torch.take_along_dim(token_probs, start_indices, dim=1)  # (batch_size, 1, 1)
        return word_probs


@register_prob_aggregator("word_boundary")
class WordBoundaryProbabilityAggregator(ChainProbabilityAggregator):
    """
    Aggregator that computes the probability of a word following the methodology
    outlined by `Pimentel and Meister (2024) <https://aclanthology.org/2024.emnlp-main.1020/>`_
    which distinguishes cases where subword tokens form a complete word from cases where
    they serve as prefixes requiring continuation.
    """
    def __init__(self, tgt_dict: Dict) -> None:
        """
        Initializes a mask that identifies all tokens in the model's target vocabulary that correspond
        to beginning-of-word tokens. This mask is used to calculate the probability of entire words based
        on the probabilities of their constituent subword tokens (e.g., the word "stata" composed of the
        subword tokens "▁sta" and "ta"). It distinguishes cases where these tokens form a complete word
        from cases where they serve as prefixes requiring continuation. This approach follows the methodology
        outlined by `Pimentel and Meister (2024) <https://aclanthology.org/2024.emnlp-main.1020/>`_
        Args:
            - tgt_dict: target dictionary containing the vocabulary of the model.
        """
        # Boolean tensor of size (vocab_size) that is True if the token is a beginning-of-word token
        self.bow_tokens_mask = torch.zeros(len(tgt_dict), dtype=torch.bool)
        bow_symbols = set(string.punctuation)
        # We remove the apostrophe because it should belong to the word that precedes it
        # e.g. in "▁un ' ▁artista", the words are "▁un '" and "▁artista"
        bow_symbols.remove("'")
        bow_symbols.add("\u2581")   # bow symbol
        bow_symbols.add(tgt_dict.eos())
        for tok, index in tgt_dict.indices.items():
            if tok[0] in bow_symbols:
                self.bow_tokens_mask[index] = True

    def _get_bow_probs(self, probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability of starting a new word at a given point in the generation process
        (indicated by the `indices` argument).
        Args:
            - probs: Tensor of size (batch_size, seq_len, vocab_size)
            - indices: Tensor of integers of size (batch_size) containing the index of the output 
                       probability distribution we want to look at in each utterance of the batch.
        Returns:
            - Tensor of size (batch_size, 1, 1) containing the sum of the probabilities corresponding
              to beginning-of-word tokens in the relevant probability distribution 
              (typically, output before or after the generation of the gender term).
        """
        indices = indices.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
        bow_probs = probs.take_along_dim(indices, dim=1)  # (batch_size, 1, vocab_size)
        if probs.device != self.bow_tokens_mask.device:
            self.bow_tokens_mask = self.bow_tokens_mask.to(probs.device)
        bow_probs = bow_probs[:, :, self.bow_tokens_mask]  # (batch_size, 1, bow_size')
        return bow_probs.sum(dim=2, keepdim=True)  # (batch_size, 1, 1)

    def compute_word_probability(
            self,
            probs: torch.Tensor,
            tgt_tokens: torch.LongTensor,
            start_indices: torch.Tensor,
            end_indices: torch.Tensor) -> torch.Tensor:
        """
        Computes the word probability integrating the methodology by 
        `Pimentel and Meister (2024) <https://aclanthology.org/2024.emnlp-main.1020/>`_
        Args:
            - probs:    The tensor containing the probability distributions over the vocabulary
                output by the model (B x S x V).
            - tgt_tokens:   The tensor containing the tokens predicted (B x S).
            - start_indices:    Tensor of length B containing the indices of the start of the word of interest.
            - end_indices:  Tensor of length B containing the indices of the end of the word of interest.
        Returns:
            - The probability of the word of interest for each item in the batch (B x 1 x 1).
        """
        # Get word level probabilities using the chain rule
        chain_rule_probs = super().compute_word_probability(probs, tgt_tokens, start_indices, end_indices)
        
        # Scale the word level probability according to the methodology by Pimentel and Meister 2024
        indices_after = end_indices + 1 
        # +1 because we want the probability distribution of the generation step after the word of interest
        seq_len = probs.size(1)
        assert 0 <= torch.all(indices_after) and torch.all(indices_after < seq_len), \
            "End indices must be within the sequence length - 1: a well-formed sequence should end with </s>, not a word."
        bow_prob_after = self._get_bow_probs(probs, indices_after)
        bow_prob_before = self._get_bow_probs(probs, start_indices)
        final_word_probs = chain_rule_probs * bow_prob_after / bow_prob_before

        return final_word_probs
