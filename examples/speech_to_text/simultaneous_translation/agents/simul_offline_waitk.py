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

import itertools

import torch
import torch.nn.functional as F

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import FairseqSimulSTAgent, BOW_PREFIX

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class WaitkAgent(FairseqSimulSTAgent):
    def __init__(self, args):
        super().__init__(args)
        self.waitk = args.waitk
        torch.set_grad_enabled(False)

    @staticmethod
    def add_args(parser):
        # fmt: off
        FairseqSimulSTAgent.add_args(parser)
        parser.add_argument("--waitk", type=int, default=None,
                            help="Wait k lagging value for test.")
        parser.add_argument("--adaptive-segmentation", default=False, action="store_true",
                            help="Whether to enable CTC prediction (greedy) to identify the number of words contained "
                                 "in the speech.")
        parser.add_argument("--vocabulary-type", type=str, choices=["sentencepiece", "subwordnmt"], default=None,
                            help="If adaptive segmentation is used, the vocabulary type has to be specified")

        # fmt: on
        return parser

    def load_model_vocab(self, args):
        super().load_model_vocab(args)
        # Check vocabulary type
        if self.args.adaptive_segmentation:
            assert self.args.vocabulary_type is not None
        self.ctc_blank_idx = self.model.encoder.dictionary.index("<ctc_blank>")

    def initialize_states(self, states):
        super().initialize_states(states)
        states.n_predicted_words = 0
        states.n_audio_words = 0
        states.full_pred = None

    def update_model_encoder(self, states):
        super().update_model_encoder(states)
        states.n_audio_words = self.get_audio_words(states)

    def get_audio_words(self, states):
        ctc_pred_greedy = F.log_softmax(states.encoder_states["ctc_out"], dim=-1).transpose(0, 1).max(dim=-1)[1]
        ctc_pred_greedy_unique = [k for k, _ in itertools.groupby(ctc_pred_greedy.tolist()[0])]
        audio_words = self.srcdict.string(
            [tok for tok in ctc_pred_greedy_unique if tok != self.ctc_blank_idx],
            bpe_symbol=self.args.vocabulary_type
        )
        return len(audio_words.split(" ")) if audio_words != "" else 0

    def _select_words(self, new_hypo, selected_n_words):
        selected_idxs = []
        curr_n_words = 0
        for idx in new_hypo:
            if self.tgtdict[idx].startswith(BOW_PREFIX):
                curr_n_words += 1
                if curr_n_words > selected_n_words:
                    break
            selected_idxs.append(idx)
        return selected_idxs

    def new_hypo(self, states):
        states.new_segment = False
        prefix_tokens = self._get_prefix(states)
        hypo = self.generate_hypothesis(states, prefix_tokens)
        hypo = hypo['tokens'].int().cpu()
        new_hypo = hypo[self._get_prefix_len(prefix_tokens):]
        return new_hypo

    def waitk_prediction(self, states):
        new_hypo = self.new_hypo(states)
        selected_n_words = states.n_audio_words - (states.n_predicted_words + self.waitk)
        states.n_predicted_words += selected_n_words
        selected_idxs = self._select_words(new_hypo, selected_n_words)

        if selected_idxs:
            states.write = selected_idxs
            return True
        return False

    def _emit_remaining_tokens(self, states):
        states.write = self.new_hypo(states)

    def _policy(self, states):
        """
        It generates the translation hypothesis starting from the encoder states
        contained in *states.encoder_states* and verify if the number of
        predicted words minus the number of the emitted ones is greater than the
        k value and, in case this is verified, all the words but the last k ones
        are emitted.
        """
        # Number of words for the wait-k policy
        n_words = states.n_predicted_words + self.waitk

        if self.args.adaptive_segmentation:
            # Adaptive segmentation based on the CTC prediction (greedy)
            if states.n_audio_words > n_words and self.waitk_prediction(states):
                return WRITE_ACTION
        else:
            # Fixed segmentation based on a fixed number of encoder states
            n_audio_words = states.encoder_states["ctc_lengths"].item() // self.args.speech_segment_factor
            if n_audio_words > n_words and self.waitk_prediction(states):
                return WRITE_ACTION
        return READ_ACTION



