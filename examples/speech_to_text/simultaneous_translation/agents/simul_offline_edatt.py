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
import torch.nn.functional as F

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import FairseqSimulSTAgent, \
    TensorListEntry

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class EDAttSTAgent(FairseqSimulSTAgent):
    """
    EDAtt policy for Simultaneous Speech Translation based on
    "Attention as a Guide for Simultaneous Speech Translation"
    (https://arxiv.org/abs/2212.07850)
    by Papi et al., 2022.
    The agent guides the simultaneous inference by exploiting the attention scores extracted from
    the decoder layer *extract_attn_from_layer*: for each token, it sums the attention scores
    of the last *frame_num* frames and verify if it exceeds *attn_threshold*. If this condition
    is verified, the emission at the current time step is stopped, otherwise we proceed to the
    next token.
    The implementation works only for SentencePiece up to now.
    """
    def __init__(self, args):
        super().__init__(args)
        self.attn_threshold = args.attn_threshold
        self.frame_num = args.frame_num
        self.extract_attn_from_layer = args.extract_attn_from_layer
        torch.set_grad_enabled(False)

    @staticmethod
    def add_args(parser):
        # fmt: off
        FairseqSimulSTAgent.add_args(parser)
        parser.add_argument("--attn-threshold", type=float, default=0.1,
                            help="Threshold on the attention scores that triggers READ action."
                                 "If the last frame attention score >= attn_threshold, READ action is performed.")
        parser.add_argument("--extract-attn-from-layer", default=4, type=int,
                            help="Layer to which extract attention scores."
                                 "Please notice that Layer 1 corresponds to 0.")
        parser.add_argument("--frame-num", default=0, type=int,
                            help="Number of frames to consider for the attention scores starting from the end.")
        # fmt: on
        return parser

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()
        states.new_segment = False
        states.write = []

    def _generate_hypothesis(self, sample, prefix_tokens, states):
        with torch.no_grad():
            hypos = self.generator.generate(
                self.model, sample, prefix_tokens=prefix_tokens, pre_computed_encoder_outs=[states.encoder_states],
                extract_attn_from_layer=self.extract_attn_from_layer
            )
        return hypos

    def _predict(self, states):
        """
        This method takes *states* as input, generates a translation hypothesis, and returns the most likely
        hypothesis *hypo* together with the length of the previously generated tokens *prefix_len*.
        """
        states.new_segment = False
        previous_output = torch.tensor([states.units.target.value], dtype=torch.int64)
        hypo, prefix_tokens = self.generate_hypothesis(states, previous_output)
        prefix_len = len(prefix_tokens[0]) if prefix_tokens is not None else 0
        return hypo, prefix_len

    def check_attention_threshold(self, hypo, prefix_len, states):
        """
        This method takes as input the generated hypothesis *hypo*, the length of the previous output tokens
        *prefix_len*, and the states *states*. It normalizes the attention scores considering all but last
        attention scores, sums the last *self.frame_num* attention scores and checks if this sum exceeds the
        attention threshold *self.attn_threshold*
        """
        hypo_tokens = hypo['tokens'].int().cpu()
        hypo_attn = hypo['attention'].float().cpu()
        # select new partial hypothesis (without the already emitted tokens)
        new_hypo = hypo_tokens[prefix_len:]
        # Remove prefix_token_idx attention score in case of multilingual generation
        # prefix_token_idx is already removed from the hypo_tokens because is counted in the prefix_len
        if self.prefix_token_idx:
            hypo_attn = hypo_attn[:, 1:]
        # normalize considering all but last attention score
        normalized_attn = F.normalize(hypo_attn[:-1, :].transpose(0, 1), dim=1)
        if normalized_attn.shape[0] > 0 and normalized_attn.shape[1] > 0:
            # find which tokens rely on the last n frames using threshold
            curr_attn = normalized_attn[prefix_len:, -self.frame_num:]
            last_frames_attn = torch.sum(curr_attn, dim=1)
            # for each element of the tensor, we check if the sum exceeds or is equal to the attn_threshold,
            # we find the list of indexes for which this is True (corresponding to 1 value) by applying
            # the nonzero() function, and we select the first token for which the threshold has been
            # exceeded, corresponding to the index from which the emission is stopped
            invalid_token_idxs = (last_frames_attn >= self.attn_threshold).nonzero(as_tuple=True)[0]
            if len(invalid_token_idxs) > 0:
                new_hypo = new_hypo[:invalid_token_idxs[0]]
            if len(new_hypo) > 0:
                states.write = new_hypo
                return True
        return False

    def valid_prediction(self, states):
        """
        This method takes *states* as input, generates the translation hypothesis
        (also containing the attention scores) and verify if at least one token can be emitted.
        """
        hypo, prefix_len = self._predict(states)
        return self.check_attention_threshold(hypo, prefix_len, states)

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION

        if len(states.write) > 0:
            return WRITE_ACTION
        if states.finish_read():
            # finish writing the hypo
            hypo, prefix_len = self._predict(states)
            hypo_tokens = hypo['tokens'].int().cpu()
            states.write = hypo_tokens[prefix_len:]
            return WRITE_ACTION
        if states.new_segment and self.valid_prediction(states):
            return WRITE_ACTION
        return READ_ACTION
