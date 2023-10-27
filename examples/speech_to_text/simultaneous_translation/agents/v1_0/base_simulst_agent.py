# Copyright 2022 FBK

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

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class TensorListEntry(ListEntry):
    """
    Data structure to store a list of tensor.
    """

    def append(self, value):
        if len(self.value) == 0:
            self.value = value
            return
        self.value = torch.cat([self.value] + [value], dim=0)

    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": len(self),
            "value": "" if type(self.value) is list else self.value.size(),
        }


class FairseqSimulSTAgent(SpeechAgent, BaseSimulSTAgent):
    """
    Base agent for Simultaneous Speech Translation which works with SimulEval<=1.0.2.
    It includes generic methods to:
    - build states (*build_states*) and initialize (*initialize_states*) in which the useful
    information that as to be maintained through time steps are memorized and updated at each
    READ_ACTION through *update_states_read*;
    - transform the audio information to features (*segment_to_units*) and the generated token indexes
    to full and detokenized words (*units_to_segment*); only SentencePiece is supported;
    - integrates the logic of the policy (*policy*) both in the case in which the input is incrementally
    received (*_policy*) and in the case in which the input has been completely received (*_emit_tokens*)
    - pass to SimulEval the indexes of the tokens to predict (*predict*)
    
    To be used, implement (override) *_emit_tokens* and *_policy*.
    """
    speech_segment_size = 40  # in ms, 4 pooling ratio * 10 ms step size

    def __init__(self, args):
        super().__init__(args)
        BaseSimulSTAgent.__init__(self, args)

        # to obtain the final speech segment size, the speech segment unit defined in
        # "speech_segment_size" is multiplied by "speech_segment_factor" which is a
        # hyper-parameter that controls the final speech segment dimension
        self.speech_segment_size *= args.speech_segment_factor

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = SpeechStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()
        states.write = []
        states.new_segment = False

    def segment_to_units(self, segment, states):
        # Convert speech samples to features
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, units, states):
        # Merge sub word to full word
        if self.eos_idx == units[0]:
            return DEFAULT_EOS

        segment = []
        for index in units:
            if index is None:
                units.pop()
            if self.prefix_token_idx is not None and index == self.prefix_token_idx:
                units.pop()
            token = self.tgtdict[index]
            if token.startswith(BOW_PREFIX) or token == DEFAULT_EOS:
                if len(segment) == 0:
                    if token != DEFAULT_EOS:
                        segment.append(token.replace(BOW_PREFIX, ""))
                    else:
                        segment.append(DEFAULT_EOS)
                else:
                    # Remove the already processed units from the SimulEval queue
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]

                    if self.eos_idx == units[0]:
                        string_to_return.append(DEFAULT_EOS)

                    return string_to_return
            else:
                segment.append(token.replace(BOW_PREFIX, ""))

        if len(units) > 0 and self.eos_idx == units[-1] or len(states.units.target) > self.max_len:
            tokens = self.model.decoder.dictionary.string([unit for unit in units if unit != DEFAULT_EOS])
            return [tokens.replace(BOW_PREFIX, ""), DEFAULT_EOS]
        # Sending None avoids any update to the scores (e.g., by returning an empty string
        # a new delay/latency score is incorrectly added by SimulEval)
        return None

    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0))
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)]))
        states.encoder_states = self.model.encoder(src_indices, src_lengths)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action
        if not states.finish_read():
            self.update_model_encoder(states)
            states.new_segment = True

    def _get_prefix(self, states):
        if states.units.target.value:
            prefix_tokens = torch.LongTensor([states.units.target.value])
            if self.prefix_token_idx is not None:
                return torch.cat(
                    (torch.LongTensor([[self.prefix_token_idx]]), prefix_tokens), dim=1)
            return prefix_tokens
        else:
            if self.prefix_token_idx is not None:
                return torch.LongTensor([[self.prefix_token_idx]])
            return None

    def generate_hypothesis(self, states, prefix_tokens):
        """
        This method takes *states* and *prefix_tokens* as inputs and generates and returns the mostly likely
        translation hypothesis *hypo*.
        """
        sample = {
            'net_input': {
                'src_tokens': self.to_device(states.units.source.value.unsqueeze(0)),
                'src_lengths': self.to_device(torch.LongTensor([states.units.source.value.size(0)]))
            }
        }
        prefix_tokens = self.to_device(prefix_tokens) if prefix_tokens is not None else None
        hypos = self._generate_hypothesis(sample, prefix_tokens, states)
        return hypos[0][0]  # We consider only the most likely hypothesis

    @staticmethod
    def _get_prefix_len(prefix):
        return len(prefix[0]) if prefix is not None else 0

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION

        # Write the remaining generated hypothesis
        if len(states.write) > 0:
            return WRITE_ACTION

        # If the input has been completely received, write the remaining hypothesis
        if states.finish_read():
            self._emit_remaining_tokens(states)
            return WRITE_ACTION

        # If no new input is received, read more
        if not states.new_segment:
            return READ_ACTION

        return self._policy(states)

    def predict(self, states):
        """
        Consumes and returns the idx(s) in *states.write* one at a time.
        """
        if len(states.write) == 0:
            return self.eos_idx
        idx_to_write = states.write[0]
        states.write = states.write[1:]
        return idx_to_write
