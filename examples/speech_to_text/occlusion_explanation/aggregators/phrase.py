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
from string import punctuation
from typing import Tuple, Dict, Union, List

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.aggregators import \
    register_aggregator, Aggregator
from examples.speech_to_text.occlusion_explanation.aggregators.utils import _min_max_normalization, \
    _mean_std_normalization


SPACE_TAG = "\u2581"  # special token used by SentencePiece for space


@register_aggregator("phrase_and_token")
class PhraseLevelAggregator(Aggregator):
    """
    Aggregates the heatmaps of filterbanks and target embeddings by averaging them
    over multiple specified tokens. Dimension 1 of target embeddings represents tokens.

    For filterbank explanations, which have a size of (n_tokens, time, channels/1),
    the aggregation is performed over the first dimension (dimension 0), representing
    the sequence length. This transforms the size to (n_phrases, time, channels/1).
    For target embedding explanations, which have a size of (n_tokens, n_tokens,
    embeddings/1), the aggregation is performed over the first dimension (dimension 0),
    at the level of words or higher-level units (broadly defined as phrases). This
    results in a size of (n_phrases, n_tokens, embeddings/1).
    """
    @staticmethod
    def get_words(tokens: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Given the list of tokens for a sample, returns a list of words (with punctuation
        considered as separate elements) and a list of tuples, where each tuple contains
        the starting and ending indices which map the words to the token positions.

        Warning: this method works properly only for SentencePiece tokenization, where
        spaces are tagged with `▁` i.e. \u2581 (unicode for "Lower One Eighth Block").
        Moreover, it splits punctuation (e.g., ["I'm"] becomes ["I", "'", "m"].
        """
        words = []
        word_indices = []
        current_word = ""
        start_index = 0
        for i, token in enumerate(tokens):
            if token.strip(SPACE_TAG) in punctuation:  # token is punctuation
                if current_word:
                    words.append(current_word)
                    word_indices.append((start_index, i - 1))
                    current_word = ""
                words.append(token.strip(SPACE_TAG))
                word_indices.append((i, i))

            elif token.startswith(SPACE_TAG):  # token is at beginning position
                if current_word:
                    words.append(current_word)
                    word_indices.append((start_index, i - 1))
                current_word = token[1:]  # remove the marker
                start_index = i

            elif current_word:  # token is in the middle/end of a word
                current_word += token

            else:  # token is after a punctuation mark or before the first word
                words.append(token)
                word_indices.append((i, i))

        # add the last word
        if current_word:
            words.append(current_word)
            word_indices.append((start_index, len(tokens) - 1))

        return words, word_indices

    @staticmethod
    def _normalize(fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Identity function, does not perform any normalization.
        """
        return fbank_map, tgt_map

    def aggregate_fbank_explanations(
            self,
            aggregation_indices: List[Tuple[int, int]],
            fbank_heatmap: Tensor,
            include_eos: bool = False) -> Tensor:
        """
        Aggregates filterbank explanations by averaging over tokens belonging to the same phrase.
        """
        # if first indices span is related to </s>, start from the subsequent
        aggregation_indices = aggregation_indices[1:] if include_eos else aggregation_indices

        stack = []
        for start, end in aggregation_indices:
            assert end >= start, "Invalid indices, 'start' is greater than 'end'."
            stack.append(fbank_heatmap[start - 1:end].mean(dim=0))

        # if first indices span is related to </s>, add its explanation
        if include_eos:
            stack.append(fbank_heatmap[-1].squeeze(0))

        return torch.stack(stack)

    def aggregate_tgt_explanations_rows(
            self,
            aggregation_indices: List[Tuple[int, int]],
            tgt_embed_heatmap: Tensor,
            include_eos: bool = False) -> Tensor:
        """
        Aggregates target embedding explanations over the first dimension according
        to the indices spans provided (`aggregation_indices`).
        """
        # if first indices span is related to </s>, start from the subsequent span for dimension 0,
        # but not for dimension 1
        aggregation_indices = aggregation_indices[1:] if include_eos else aggregation_indices
        stack = []
        # iterate over rows (dimension 0)
        for row_ind_start, row_ind_end in aggregation_indices:
            assert row_ind_end >= row_ind_start, "Invalid indices, 'start' is greater than 'end'."
            averaged_tgt = tgt_embed_heatmap[row_ind_start - 1:row_ind_end].mean(dim=0)
            # pad from the current word onwards to exclude the first token of the word from the explanation.
            averaged_tgt[row_ind_start:] = 0
            stack.append(averaged_tgt)

        # add last explanations for </s>
        if include_eos:
            stack.append(tgt_embed_heatmap[-1])

        return torch.stack(stack)

    def __call__(
            self,
            explanations: Dict[int, Dict[str, Union[Tensor, str, List[str]]]],
            indices: Dict[int, Tuple[List[str], List[Tuple[int, int]]]] = None
    ) -> Dict[int, Dict[str, Union[Tensor, List[str]]]]:
        """
        Performs aggregation over multiple specified tokens.
        The tokens to aggregate are specified in `indices` argument, which is a
        dictionary mapping each sample_id to a tuple, composed of a list that contains
        the aggregated tokens, and  list of tuples, where each tuple contains the
        starting and ending indices of the tokens which compose the desired phrase.
        If `indices` is None, it aggregates all the tokens containing subwords into
        words using the `get_words()` method.

        Target embedding explanations have quadratic forms in the first two dimensions:
        (seq_len, seq_len, embedding_size/1). These two dimensions, however, are shifted by one
        position: positions in dimension 0 represents the token which is explained, while, positions
        in dimension 1 represents the token which are fed to predict the token which is explained.
        For example, if the target text is ["<\s>", "_I", "_am", "_a", "_research", "er"] the
        explanation matrix looks like
                                            token fed
        token explained         <\s>    _I  _am _a  _research   er
        _I                      1       0   0   0   0           0
        _am                     1       1   0   0   0           0
        _a                      1       1   1   0   0           0
        _research               1       1   1   1   0           0
        er                      1       1   1   1   1           0
        <\s>                    1       1   1   1   1           1

        If explanation for the final token </s> is required, (0, 0) needs to be added to the
        indices spans and the row considering the all the previous (aggregated) tokens is provided.
        """
        for sample_id in list(explanations.keys()):
            if indices:
                try:
                    phrases, aggregation_indices = indices[sample_id]
                except KeyError:
                    raise KeyError(f"Sample_id {sample_id} not available in provided indices.")
            else:
                tgt_text = explanations[sample_id]["tgt_text"]
                phrases, aggregation_indices = self.get_words(tgt_text)

            assert len(phrases) == len(aggregation_indices), \
                f"Size of indices ({len(aggregation_indices)}) does not correspond to the size " \
                f"of phrases ({len(phrases)})."

            if aggregation_indices:  # empty tuple means that no aggregation is performed
                fbank_heatmap = explanations[sample_id]["fbank_heatmap"]
                tgt_embed_heatmap = explanations[sample_id]["tgt_embed_heatmap"]

                # normalize explanations
                fbank_heatmap, tgt_embed_heatmap = self._normalize(fbank_heatmap, tgt_embed_heatmap)

                include_eos = True if (0, 0) in aggregation_indices else False
                aggregated_fbank_explanations = self.aggregate_fbank_explanations(
                    aggregation_indices, fbank_heatmap, include_eos)
                aggregated_tgt_explanations = self.aggregate_tgt_explanations_rows(
                    aggregation_indices, tgt_embed_heatmap, include_eos)

                explanations[sample_id]["fbank_heatmap"] = aggregated_fbank_explanations
                explanations[sample_id]["tgt_embed_heatmap"] = aggregated_tgt_explanations
                explanations[sample_id]["tgt_phrases"] = phrases
            else:
                del explanations[sample_id]

        return explanations


@register_aggregator("phrase_and_token_min_max_norm")
class PhraseLevelAggregatorMinMaxNormalization(PhraseLevelAggregator):
    """
    Aggregates the heatmaps of filterbanks and target embeddings by averaging them
    over multiple specified tokens. Dimension 1 of target embeddings represents tokens.
    Before averaging, token-level min-max normalization between explanations of
    filterbanks and of target embeddings is applied.
    """
    @staticmethod
    def _normalize(fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        return _min_max_normalization(fbank_map, tgt_map)


@register_aggregator("phrase_and_token_mean_std_norm")
class PhraseLevelAggregatorMeanStdNormalization(PhraseLevelAggregator):
    """
    Aggregates the heatmaps of filterbanks and target embeddings by averaging them
    over multiple specified tokens. Dimension 1 of target embeddings represents tokens.
    Before averaging, token-level mean-std normalization between explanations of
    filterbanks and of target embeddings is applied.
    """
    @staticmethod
    def _normalize(fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        return _mean_std_normalization(fbank_map, tgt_map)


@register_aggregator("phrase_and_word")
class PhraseAndWordLevelAggregator(PhraseLevelAggregator):
    """
    Aggregates the heatmaps of filterbanks and target embeddings by averaging them
    over multiple specified tokens. Dimension 1 of target embeddings represents words.

    For filterbank explanations, which have a size of (n_tokens, time, channels/1),
    the aggregation is performed over the first dimension (dimension 0), representing
    the sequence length. This transforms the size to (n_phrases, time, channels/1).
    For target embedding explanations, which have a size of (n_tokens, n_tokens,
    embeddings/1), the aggregation is performed over the first dimension (dimension 0),
    at the level of words or higher-level units (broadly defined as phrases), and over
    the second dimension, at the level of words. This results in a size of (n_phrases,
    n_words, embeddings/1). Note that if the phrases correspond to words, the size is
    (n_words, n_words, embeddings/1).
    """
    def aggregate_tgt_explanations_cols(
            self,
            aggregation_indices: List[Tuple[int, int]],
            tgt_embed_heatmap: Tensor) -> Tensor:
        """
        Aggregates target embedding explanations over the second dimension at word level.
        """
        stack = []
        # iterate over rows (dimension 1)
        for col_ind_start, col_ind_end in aggregation_indices:
            stack.append(
                tgt_embed_heatmap[:, col_ind_start:col_ind_end + 1].mean(dim=1, keepdim=True))
        return torch.cat(stack, dim=1)

    def __call__(
            self,
            explanations: Dict[int, Dict[str, Union[Tensor, str, List[str]]]],
            indices: Dict[int, Tuple[List[str], List[Tuple[int, int]]]] = None
    ) -> Dict[int, Dict[str, Tensor]]:
        """
        Performs aggregations over both dimension 0 and dimension 1.
        The tokens to aggregate in the first dimension are specified in `indices` argument,
        which is a dictionary mapping each sample_id to a tuple, composed of a list that
        contains the aggregated tokens, and  list of tuples, where each tuple contains the
        starting and ending indices of the tokens which compose the desired phrase.
        If `indices` is None, it aggregates all the tokens containing subwords into
        words using the `get_words()` method.
        The tokens to aggregate in the second dimension correspond to the single words.

        Target embedding explanations have quadratic forms in the first two dimensions:
        (seq_len, seq_len, embedding_size/1). These two dimensions, however, are shifted by one
        position: positions in dimension 0 represents the token which is explained, while, positions
        in dimension 1 represents the token which are fed to predict the token which is explained.
        For example, if the target text is ["<\s>", "_I", "_am", "_a", "_research", "er"] the
        explanation matrix looks like
                                            token fed
        token explained         <\s>    _I  _am _a  _research   er
        _I                      1       0   0   0   0           0
        _am                     1       1   0   0   0           0
        _a                      1       1   1   0   0           0
        _research               1       1   1   1   0           0
        er                      1       1   1   1   1           0
        <\s>                    1       1   1   1   1           1

        If explanation for the final token </s> is required, (0, 0) needs to be added to the
        indices spans and the row considering the all the previous (aggregated) tokens is provided.
        """
        aggregated_explanations = super().__call__(explanations, indices)
        for sample_id, explanation_data in aggregated_explanations.items():
            phrases, aggregation_indices = self.get_words(explanation_data["tgt_text"])
            explanation_data["tgt_embed_heatmap"] = self.aggregate_tgt_explanations_cols(
                aggregation_indices, explanation_data["tgt_embed_heatmap"])
            explanation_data["tgt_text"] = phrases
        return aggregated_explanations


@register_aggregator("phrase_and_word_min_max_norm")
class PhraseAndWordLevelAggregatorMinMaxNormalization(PhraseAndWordLevelAggregator):
    """
    Aggregates the heatmaps of filterbanks and target embeddings by averaging them
    over multiple specified tokens. Dimension 1 of target embeddings represents words.
    Before averaging, token-level min-max normalization between explanations of
    filterbanks and of target embeddings is applied.
    """
    @staticmethod
    def _normalize(fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        return _min_max_normalization(fbank_map, tgt_map)


@register_aggregator("phrase_and_word_mean_std_norm")
class PhraseAndWordLevelAggregatorMeanStdNormalization(PhraseAndWordLevelAggregator):
    """
    Aggregates the heatmaps of filterbanks and target embeddings by averaging them
    over multiple specified tokens. Dimension 1 of target embeddings represents words.
    Before averaging, token-level mean-std normalization between explanations of
    filterbanks and of target embeddings is applied.
    """
    @staticmethod
    def _normalize(fbank_map: Tensor, tgt_map: Tensor) -> Tuple[Tensor, Tensor]:
        return _mean_std_normalization(fbank_map, tgt_map)
