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

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import FairseqSimulSTAgent, \
    TensorListEntry, BOW_PREFIX

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class WaitkAgent(FairseqSimulSTAgent):
    def __init__(self, args):
        super(FairseqSimulSTAgent).__init__(args)
        torch.set_grad_enabled(False)

    @staticmethod
    def add_args(parser):
        # fmt: off
        FairseqSimulSTAgent.add_args(parser)
        parser.add_argument("--waitk", type=int, default=None,
                            help="Wait k lagging value for test.")
        parser.add_argument("--speech-segment-factor", type=int, default=1,
                            help="Factor multiplied by speech segment size of 40ms to obtain the final speech segment "
                                 "size used in the adaptive module.")
        parser.add_argument("--adaptive-segmentation", default=False, action="store_true",
                            help="Whether to enable CTC prediction (greedy) to identify the number of words contained "
                                 "in the speech.")
        parser.add_argument("--vocabulary-type", type=str, choices=["sentencepiece", "subwordnmt"], default=None,
                            help="If adaptive segmentation is used, the vocabulary type has to be specified")

        # fmt: on
        return parser

    def load_model_vocab(self, args):
        super(FairseqSimulSTAgent).load_model_vocab(args)
        # Check vocabulary type
        if self.args.adaptive_segmentation:
            assert self.args.vocabulary_type is not None
        self.ctc_blank_idx = self.model.encoder.dictionary.index("<ctc_blank>")

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()
        states.prev_toks = []
        states.to_predict = []
        states.n_predicted_words = 0
        states.n_audio_words = 0
        states.full_pred = None

    def update_model_encoder(self, states):
        super(FairseqSimulSTAgent).update_model_encoder(states)
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

    def waitk_prediction(self, states):
        prefix_tokens = self.to_device(self._get_prefix(states)) if self._get_prefix(states) is not None else None
        hypo_tokens, _ = FairseqSimulSTAgent.generate_hypothesis(states, prefix_tokens)

        states.full_pred = hypo_tokens
        new_hypo = hypo_tokens[len(states.prev_toks):]
        selected_n_words = states.n_audio_words - (states.n_predicted_words + self.args.waitk)
        states.n_predicted_words += selected_n_words
        selected_idxs = self._select_words(new_hypo, selected_n_words)

        torch.cuda.empty_cache()

        if selected_idxs:
            states.to_predict = selected_idxs
            states.prev_toks += selected_idxs
            return True
        else:
            states.prev_toks += selected_idxs
            return False

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION

        if len(states.to_predict) > 0:
            return WRITE_ACTION

        # Set a maximum to avoid possible loops by the system
        if states.n_predicted_words > self.args.max_len:
            states.status['write'] = False

        # Number of words for the wait-k policy
        n_words = states.n_predicted_words + self.args.waitk

        action = 0
        # If finish_read, write the remaining output
        if states.finish_read():
            if states.full_pred is None:
                states.full_pred = self._predict(states)
            final_hypo = states.full_pred[len(states.prev_toks):]
            states.to_predict = final_hypo
            return WRITE_ACTION

        if self.args.adaptive_segmentation:
            # Adaptive segmentation based on the CTC prediction (greedy)
            if states.n_audio_words > n_words:
                action = 1
        else:
            # Fixed segmentation
            # Take an action based on the number of encoder states
            if states.encoder_states["ctc_lengths"].item() // self.args.speech_segment_factor > n_words:
                action = 1

        if action == 0:
            return READ_ACTION
        else:
            if self.waitk_prediction(states):
                return WRITE_ACTION
            else:
                return READ_ACTION

    def predict(self, states):
        idx = states.to_predict[0]
        states.to_predict = states.to_predict[1:]
        return idx
