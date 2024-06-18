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

from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_alignatt import AlignAttSTAgent, EDAttSTAgent
from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t import SimulSeamlessS2T

from simuleval.agents import AgentStates, WriteAction, ReadAction
from simuleval.utils import entrypoint

from typing import Optional


@entrypoint
class AlignAttSeamlessS2T(SimulSeamlessS2T):
    """
    The SimulST agent based on SeamlessM4T that implements the
    `AlignAtt policy <https://www.isca-archive.org/interspeech_2023/papi23_interspeech.html>`_.
    """
    def __init__(self, args):
        super().__init__(args)
        self.frame_num = args.frame_num
        self.extract_attn_from_layer = args.extract_attn_from_layer

    @staticmethod
    def add_args(parser):
        SimulSeamlessS2T.add_args(parser)
        EDAttSTAgent.add_edatt_args(parser)

    def _policy(self, states: Optional[AgentStates] = None):
        input_features = self.get_input_features(states)
        prefix_ids = self.get_prefix(states)
        new_hypo, hypo_attn = self.generate_with_cross_attn(
            input_features, prefix_ids, self.extract_attn_from_layer)

        if len(new_hypo) > 0:
            new_hypo = AlignAttSTAgent.alignatt_policy(
                new_hypo=new_hypo, hypo_attn=hypo_attn, frame_num=self.frame_num)
            valid_words = self.get_words(self.tokenizer.convert_ids_to_tokens(new_hypo))
            if valid_words:
                return WriteAction(
                    content=valid_words,
                    finished=len(states.target) > self.max_len or states.source_finished)
        return ReadAction()
