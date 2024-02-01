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
import logging
from typing import List

import torch

from examples.speech_to_text.simultaneous_translation.agents.speech_utils import BOW_PREFIX
from examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.history_selection import HistorySelection
from fairseq.data import Dictionary
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from simuleval.agents import Action, AgentStates


logger = logging.getLogger(__name__)

SUBSAMPLING_FACTOR = 4


class AudioAttentionHistorySelectionBase(HistorySelection):
    """
    Audio history selection method based on attention mechanism.
    The selection is performed based on the encoded audio features, stored in *states.source*, the
    *current_hypo*, which is added to the *states* by the SimulST agent, and the prefix with the
    new (partial) hypothesis, stored in *states.target_indices*.
    The history for the next decoding step is defined as follows:
     - First, the new textual history is selected by the textual history selection method, which is
     in charge of selecting the tokens to retain;
     - Second, the new audio history is selected according to cross-attention scores between the
     audio features and the retained textual history by discarding past features that are not
     attended by any tokens of the textual history.
    """
    def __init__(self, tgt_dict: Dictionary, args):
        self.tgt_dict = tgt_dict
        self.args = args
        self.prefix_token_idx = None
        if getattr(self.args, "prefix_token", "") != "":
            lang_tag = SpeechToTextDataset.LANG_TAG_TEMPLATE.format(self.args.prefix_token)
            self.prefix_token_idx = self.tgt_dict.index(lang_tag)

    def __call__(self, action: Action, states: AgentStates):
        new_text_history = self.text_history(action, states)
        new_audio_history = self.audio_history(action, states, new_text_history)
        states.target_indices = new_text_history
        states.source = [new_audio_history]

    def text_history(self, action: Action, states: AgentStates):
        raise NotImplementedError(
            "AudioAttentionHistorySelectionBase is an abstract class and text_history must be "
            "overridden")

    def audio_history(self, action: Action, states: AgentStates, new_text_history: List[int]):
        discarded_text = len(states.target_indices) - len(new_text_history)
        if discarded_text == 0:
            return states.source[0]

        assert hasattr(states, "current_hypo"), \
            "Attention-based streaming supports only SimulST agents that store the current " \
            "hypothesis in the states"

        attn = states.current_hypo['attention']

        # Handle multilingual case
        if self.prefix_token_idx:
            attn = attn[:, 1:]

        # Select only relevant textual history
        attn = attn[:, discarded_text:]

        # Skip last attention score and end of sentence
        attn = attn[:-1, :-1].transpose(0, 1).float()

        # Compute the frame to which each token of the textual history mostly attends to
        most_attended_idxs = torch.argmax(attn, dim=1)

        # Find the first feature that is attended
        earliest_attended_idx = torch.sort(most_attended_idxs)[0][0]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Earlier attended idx: {earliest_attended_idx}")
            logger.debug(f"Audio Shape: {states.source[0].shape}")
        if 'ctc_batch_predicted' in states.current_hypo:
            # If CTC compression is applied, retrieve the original feature lengths
            earliest_attended_idx = self.convert_ctc_compressed_idx(
                states.current_hypo['ctc_batch_predicted'], earliest_attended_idx)

        # Multiply by 4 (subsampling factor of speech models) to recover the original number of frames
        frames_to_cut = earliest_attended_idx * SUBSAMPLING_FACTOR
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Frames to cut: {frames_to_cut}")

        # Cut the unattended audio features
        return states.source[0][frames_to_cut:]

    @staticmethod
    def convert_ctc_compressed_idx(ctc_predictions, idx):
        # Obtain the list containing the original length of the frames
        original_frames = [pred[1] for pred in ctc_predictions[:idx]]
        # Return the number of original frames
        return sum(original_frames)


class FixedWordsHistorySelection(AudioAttentionHistorySelectionBase):
    """
    A textual history selection method that retains a pre-defined number of words
    (*history_words*).

    The implementation works only for SentencePiece up to now.
    """
    def __init__(self, tgt_dict: Dictionary, args):
        super().__init__(tgt_dict, args)
        self.history_words = args.history_words

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--history-words", default=20, type=int,
            help="Number of history words to keep if 'fixed_words' history selection is used. "
                 "The default value is 20 since it resulted the best in the experiments.")

    def text_history(self, action: Action, states: AgentStates):
        current_text = states.target_indices

        words_to_keep = self.history_words
        new_history = []
        for idx in reversed(current_text):
            new_history.append(idx)
            # Check if 'BOW_PREFIX' (space in SentencePiece) is contained in the token,
            # meaning that we reached the beginning of the word that should be counted
            if BOW_PREFIX in self.tgt_dict[idx]:
                words_to_keep -= 1
                # When all the words to keep are consumed, the accumulation is stopped
                # and the prefix is returned
                if words_to_keep == 0:
                    break
        new_history.reverse()

        return new_history


class PunctuationHistorySelection(AudioAttentionHistorySelectionBase):
    """
    A textual history selection method that retains the sentence before the last strong punctuation
     character.

    The implementation works only for SentencePiece up to now.
    """

    STRONG_PUNCTUATION = [".", "!", "?", ":", ";"]

    def __init__(self, tgt_dict: Dictionary, args):
        super().__init__(tgt_dict, args)
        self.history_max_len = args.history_max_len

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--history-max-len", default=100, type=int,
            help="The maximum length of the textual history after which the current content is cut."
                 "Defaults to 100.")

    def text_history(self, action: Action, states: AgentStates):
        current_text = states.target_indices

        new_history = []
        for idx in reversed(current_text):
            prefix_token = self.tgt_dict[idx]
            contains_punctuation = False
            for punct in self.STRONG_PUNCTUATION:
                if punct in prefix_token:
                    contains_punctuation = True
                    break
            if contains_punctuation:
                break
            new_history.append(idx)
        new_history.reverse()

        if len(new_history) > self.history_max_len:
            # If history does not contain punctuation and is longer thant HISTORY_MAX_LENGTH
            # the textual history is cut
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"The textual history has hit the maximum predefined length of "
                    f"{self.history_max_len}")
            new_history = new_history[-self.history_max_len:]
        return new_history
