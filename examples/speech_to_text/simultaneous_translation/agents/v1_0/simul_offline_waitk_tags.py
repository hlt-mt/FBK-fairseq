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
from examples.speech_to_text.simultaneous_translation.agents.v1_0.simul_offline_waitk import WaitkAgent

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


class WaitkAgentWithTags(WaitkAgent):
    def load_model_vocab(self, args):
        super().load_model_vocab(args)
        self.tags = self.task.data_cfg.tags

    def initialize_states(self, states):
        super().initialize_states(states)
        # Store previous output tokens without considering emitted tags
        states.prev_toks = []
        states.prev_tag = 0

    def _get_prefix(self, states):
        if states.prev_toks:
            prefix_tokens = torch.tensor([states.prev_toks], dtype=torch.int64)
            if self.prefix_token_idx is not None:
                return torch.cat(
                    (torch.LongTensor([[self.prefix_token_idx]]), prefix_tokens), dim=1)
            return prefix_tokens
        else:
            if self.prefix_token_idx is not None:
                return torch.LongTensor([[self.prefix_token_idx]])
            return None

    def add_tags_to_target(self, states, hypo_tag):
        hypo_tok = states.write
        states.write = []
        for token, tag in zip(hypo_tok, hypo_tag):
            if tag != states.prev_tag:
                if states.prev_tag == 0:
                    states.write.append(torch.tensor(
                        self.tgtdict.index(f"<{self.tags[tag - 1]}>"), dtype=token.dtype))
                elif tag == 0:
                    states.write.append(torch.tensor(
                        self.tgtdict.index(f"</{self.tags[states.prev_tag - 1]}>"), dtype=token.dtype))
                else:
                    states.write.append(torch.tensor(
                        self.tgtdict.index(f"</{self.tags[states.prev_tag - 1]}>"), dtype=token.dtype))
                    states.write.append(torch.tensor(
                        self.tgtdict.index(f"<{self.tags[tag - 1]}>"), dtype=token.dtype))
            states.write.append(token)
            states.prev_tag = tag

    def new_hypo(self, states):
        states.new_segment = False
        prefix_tokens = self._get_prefix(states)
        prefix_len = self._get_prefix_len(prefix_tokens)
        hypo = self.generate_hypothesis(states, prefix_tokens)
        hypo_tokens = hypo['tokens'].int()
        new_hypo_tokens = hypo_tokens[prefix_len:]
        hypo_tags = hypo['tags'].int()
        new_hypo_tags = hypo_tags[prefix_len:]
        return new_hypo_tokens, new_hypo_tags

    def waitk_prediction(self, states):
        new_hypo, new_tags = self.new_hypo(states)
        selected_n_words = states.n_audio_words - (states.n_predicted_words + self.waitk)
        states.n_predicted_words += selected_n_words
        states.write = self._select_words(new_hypo, selected_n_words)
        if states.write:
            states.prev_toks += states.write
            new_tags = new_tags[:len(states.write)]
            if sum(new_tags != 0) > 0 or states.prev_tag != 0:
                self.add_tags_to_target(states, new_tags)
            return True
        return False

    def _emit_remaining_tokens(self, states):
        final_hypo, final_tags = self.new_hypo(states)
        states.write = final_hypo
        if sum(final_tags != 0) > 0 or states.prev_tag != 0:
            self.add_tags_to_target(states, final_tags)
        return WRITE_ACTION
