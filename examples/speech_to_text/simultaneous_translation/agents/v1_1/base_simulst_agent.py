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

import torch

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import BaseSimulSTAgent
from examples.speech_to_text.simultaneous_translation.agents.speech_utils import BOW_PREFIX
from examples.speech_to_text.simultaneous_translation.agents.v1_1.speech_utils import SpeechStates, \
    OnlineFeatureExtractorV1_1

try:
    from simuleval.agents import SpeechToTextAgent, ReadAction, WriteAction
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class FairseqSimulSTAgent(SpeechToTextAgent, BaseSimulSTAgent):
    """
    Base agent for Simultaneous Speech Translation which works with SimulEval>=1.1.0.
    It includes generic methods to:
    - build states (*build_states*) and initialize (*initialize_states*) in which the useful
    information that as to be maintained through time steps are memorized and updated at each
    ReadAction() through *update_states_read*;
    - transform the audio information into features (*get_filterbank*);
    - integrates the logic of the policy (*policy*) both in the case in which the input is
    incrementally received (*_policy*) and in the case in which the input has been completely
    received (*_emit_tokens*);
    - pass to SimulEval the indexes of the tokens to *predict* and produce the valid
    hypothesis, only SentencePiece is supported.

    To be used, the content of *_emit_tokens* and *_policy* has to be implemented.
    """

    def __init__(self, args):
        self.args = args
        assert self.source_type
        assert self.target_type
        self.device = getattr(self.args, "device", "cpu")

        BaseSimulSTAgent.__init__(self, args)

        self.to_device(self.model)

        self.feature_extractor = OnlineFeatureExtractorV1_1(args)

    def reset(self):
        """
        Reset agent, called every time when a new sentence coming in.
        """
        self.initialize_states(self.states)

    def build_states(self):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = SpeechStates(self)
        return states

    def to_device(self, tensor):
        if self.device != "cpu":
            return tensor.cuda()
        else:
            return tensor.cpu()

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.source = []
        states.target = []
        states.target_indices = []
        states.source_finished = False
        states.target_finished = False
        states.write = []
        states.new_segment = False

    def get_filterbank(self, segment):
        # Convert speech samples to features
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def update_model_encoder(self, states):
        if len(states.source) == 0:
            return
        src_indices = self.to_device(
            states.source[0].unsqueeze(0))
        src_lengths = self.to_device(
            torch.LongTensor([states.source[0].size(0)]))
        states.encoder_states = self.model.encoder(src_indices, src_lengths)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action
        if not states.source_finished:
            self.update_model_encoder(states)
            states.new_segment = True

    def _get_prefix(self, states):
        if states.target_indices:
            prefix_tokens = torch.LongTensor([states.target_indices])
            if self.prefix_token_idx is not None:
                return torch.cat(
                    (torch.LongTensor([[self.prefix_token_idx]]), prefix_tokens), dim=1)
            return prefix_tokens
        else:
            if self.prefix_token_idx is not None:
                return torch.LongTensor([[self.prefix_token_idx]])
            return None

    @staticmethod
    def _get_prefix_len(prefix):
        return len(prefix[0]) if prefix is not None else 0

    def generate_hypothesis(self, states, prefix_tokens):
        """
        This method takes *states* and *prefix_tokens* as inputs and generates and returns
        the mostly likely translation hypothesis *hypo*.
        """
        sample = {
            'net_input': {
                'src_tokens': self.to_device(states.source[0].unsqueeze(0)),
                'src_lengths': self.to_device(torch.LongTensor([states.source[0].size(0)]))
            }
        }
        prefix_tokens = self.to_device(prefix_tokens) if prefix_tokens is not None else None
        hypos = self._generate_hypothesis(sample, prefix_tokens, states)
        return hypos[0][0]  # We consider only the most likely hypothesis

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return ReadAction()

        # If the input has been completely received, write the remaining hypothesis
        if states.source_finished:
            self._emit_remaining_tokens(states)
            return WriteAction(
                self.tgtdict.string(states.write, bpe_symbol="sentencepiece"), True)

        # If no new input is received, read more
        if not states.new_segment:
            return ReadAction()

        return self._policy(states)

    def get_words(self, states):
        """
        Processes the valid predicted idx(s) in *states.write* and returns complete words
        if any, otherwise an empty string.
        """
        # Process predicted idxs and empty the states.write buffer
        idxs_to_write = states.write
        states.write = []
        # Remove last incomplete word
        num_tokens_incomplete = 1  # at least last token is always incomplete
        has_completed_words = False
        for idx in reversed(idxs_to_write):
            if self.tgtdict[idx].startswith(BOW_PREFIX):
                has_completed_words = True
                break
            else:
                num_tokens_incomplete += 1
        if has_completed_words:
            idxs_to_write = idxs_to_write[:-num_tokens_incomplete]
            for idx in idxs_to_write:
                states.target_indices.append(idx)
            return self.tgtdict.string(idxs_to_write, bpe_symbol="sentencepiece")
        return ""
