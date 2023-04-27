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

from examples.speech_to_text.simultaneous_translation.agents.speech_utils import DEFAULT_EOS
from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_edatt import EDAttSTAgent

try:
    from simuleval.agents import ReadAction, WriteAction
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class AlignAttSTAgent(EDAttSTAgent):
    def _policy(self, states):
        """
        First, it generates the translation hypothesis *hypo* (also containing the
        attention scores) and the length of the previous output tokens *prefix_len*
        starting from the encoder states contained in *states.encoder_states*.
        Second, for each token of the hypothesis, it removes the last attention
        score, finds the alignment between the token and the input frames using the
        *torch.argmax* function and checks if the corresponding aligned frame is in
        the last *self.frame_num* frames. If this condition is verified, the
        process is stopped, otherwise we proceed to the next token. If at least one
        token of the hypothesis can be emitted, the partial hypothesis is predicted
        through *self.predict*, otherwise ReadAction().
        """
        hypo, prefix_len = self._get_hypo_and_prefix(states)
        # Select new partial hypothesis (without the already emitted tokens)
        # with the relative attention scores and remove end of sentence and
        # the last attention score
        new_hypo = hypo['tokens'][prefix_len:-1].int()
        hypo_attn = hypo['attention'][:-1, prefix_len:-1].transpose(0, 1).float()
        if hypo_attn.shape[0] > 0 and hypo_attn.shape[1] > 0:
            # Compute the frame to which each token mostly attends to
            most_attended_idxs = torch.argmax(hypo_attn, dim=1)
            # Compute the first frame starting from which the model cannot attend to
            first_invalid_frame = hypo_attn.size(1) - self.frame_num
            # For each element of the tensor, corresponding to a predicted token,
            # we check if its aligned frame is equal or successive to the
            # first_invalid_frame, we find the list of indexes for which this is True
            # (corresponding to 1 value) by applying the nonzero() function, and we
            # select the first token for which an invalid alignment has been done,
            # corresponding to the index from which the emission is stopped
            invalid_tok_idxs = (most_attended_idxs >= first_invalid_frame).nonzero(
                as_tuple=True)[0]
            if len(invalid_tok_idxs) > 0:
                new_hypo = new_hypo[:invalid_tok_idxs[0]]
            # Emit the hypothesis if not empty
            if len(new_hypo) > 0:
                states.write = new_hypo
                valid_words = self.get_words(states)
                if valid_words:
                    finished = DEFAULT_EOS in valid_words or len(states.target) > self.max_len
                    return WriteAction(valid_words, finished)
        return ReadAction()
