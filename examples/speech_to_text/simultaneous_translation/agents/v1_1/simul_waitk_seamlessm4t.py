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

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents.actions import WriteAction, ReadAction

from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t import SimulSeamlessS2T


@entrypoint
class WaitkSeamlessS2T(SimulSeamlessS2T):
    """
    The SimulST agent based on SeamlessM4T that implements the
    `wait-k policy <https://aclanthology.org/P19-1289/>`_ in its adaptation to speech inputs by
    `Ma et al., (2020) <https://aclanthology.org/2020.aacl-main.58/>`_.
    """
    def __init__(self, args):
        super().__init__(args)
        self.waitk_lagging = args.waitk_lagging
        self.continuous_write = args.continuous_write

    @staticmethod
    def add_args(parser):
        SimulSeamlessS2T.add_args(parser)
        parser.add_argument("--waitk-lagging", default=1, type=int)
        parser.add_argument(
            "--continuous-write", default=1, type=int,
            help="Max number of words to write at each step")

    def _policy(self, states: Optional[AgentStates] = None):
        length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if length_in_seconds * 1000 / self.source_segment_size < self.waitk_lagging:
            return ReadAction()

        input_features = self.get_input_features(states)

        prefix_ids = self.get_prefix(states)

        prediction = self.generate_and_decode(input_features, prefix_ids)
        prediction = prediction.split()

        if self.continuous_write > 0:
            prediction = prediction[:self.continuous_write]

        return WriteAction(
            content=" ".join(prediction),
            finished=states.source_finished)
