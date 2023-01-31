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

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import FairseqSimulSTAgent, \
    TensorListEntry

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class LocalAgreementSimulSTAgent(FairseqSimulSTAgent):
    """
    Local Agreement agent for Simultaneous Speech Translation based on
    "Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection"
    (https://www.isca-speech.org/archive/pdfs/interspeech_2020/liu20s_interspeech.pdf)
    by Liu et al., 2020. The agent displays the agreeing prefixes of the two consecutive chunks:
    during the first n−1 chunks, no output is produced; from the n-th chunk on, the longest common prefix
    of the n consecutive chunks is identified and emitted.
    Empirically, the authors found that n=2 works better.
    The implementation works only for SentencePiece up to now.
    """
    def __init__(self, args):
        super().__init__(args)
        # Local Agreement using last 2 generated sentences as memory
        self.la_n = 2
        torch.set_grad_enabled(False)

    @staticmethod
    def add_args(parser):
        # fmt: off
        FairseqSimulSTAgent.add_args(parser)
        # fmt: on
        return parser

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()
        states.chunks_hyp = []
        states.displayed = []
        states.retrieved = []
        states.new_segment = False
        states.write = []

    def update_states_read(self, states):
        super().update_states_read(states)
        if not states.finish_read():
            states.new_segment = True

    def prefix(self, states):
        """
        This method takes *states* as input, which stores the hypothesis generated at each time step
        in *states.chunks_hyp*, and returns the common prefix among the last *self.la_n* hypotheses
        without including the already displayed prefix *states.displayed*.
        """
        if states.finish_read() and len(states.chunks_hyp) > 0:
            displayed = len(states.displayed)
            return states.chunks_hyp[-1][displayed:]

        if len(states.chunks_hyp) < self.la_n:
            return []

        displayed = len(states.displayed)

        prefixes = [s[displayed:] for s in states.chunks_hyp[-self.la_n:]]
        common_pref = []
        for prefix in zip(*prefixes):
            prefix_candidate = prefix[0]
            if all(prefix_el == prefix_candidate for prefix_el in prefix) and prefix_candidate != self.eos_idx:
                common_pref.append(prefix_candidate)
            else:
                break

        return common_pref

    def _get_prefix(self, states):
        if len(states.displayed) > 1 and states.displayed[0] == self.eos_idx:
            return torch.LongTensor([states.displayed[1:]])
        elif len(states.displayed) > 0:
            return torch.LongTensor([states.displayed])
        else:
            return None

    def _predict(self, states):
        """
        This method takes *states* as input, generates a translation hypothesis, and applies the *self.prefix()*
        method to obtain the common prefix among the previously generated hypotheses. It returns False if the prefix
        is empty, meaning that there is no common prefix among the generated hypotheses, and True otherwise.
        """
        states.new_segment = False
        prefix_tokens = self._get_prefix(states)
        hypo, _ = self.generate_hypothesis(states, prefix_tokens)
        hypo_tokens = hypo['tokens'].int().cpu()
        if self.prefix_token_idx:
            hypo_tokens = hypo_tokens[1:]

        states.chunks_hyp.append(hypo_tokens)
        common_pref = self.prefix(states)

        if len(common_pref) > 0:
            states.displayed.extend(common_pref)
            states.write = common_pref
            return True
        return False

    def policy(self, states):
        # Set a maximum to avoid possible loops by the system
        if len(states.units.target.value) > self.args.max_len:
            states.status['write'] = False

        if len(states.write) > 0:
            return WRITE_ACTION
        if states.new_segment and self._predict(states):
            return WRITE_ACTION
        if states.finish_read():
            # finish writing the hypo
            if self.prefix_token_idx and states.chunks_hyp[-1][0] == self.prefix_token_idx:
                states.chunks_hyp[-1] = states.chunks_hyp[-1][1:]
            states.write = states.chunks_hyp[-1][len(states.displayed):]
            return WRITE_ACTION
        return READ_ACTION

    def predict(self, states):
        if len(states.write) == 0:
            return self.eos_idx
        w = states.write[0]
        states.write = states.write[1:]
        return w
