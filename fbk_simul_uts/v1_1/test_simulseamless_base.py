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
from argparse import Namespace

from simuleval.agents import ReadAction, WriteAction

from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t import SimulSeamlessS2T

from unittest.mock import patch


class BaseSimulSeamlessTest:
    """
    It implements basic common tests to be executed for the SimulST agent based on SeamlessM4T.
    """
    def add_extra_args(self):
        pass

    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.load_model')
    def create_agent(self, mock_load_model):
        mock_load_model.return_value = None
        return SimulSeamlessS2T(self.args)

    def setUp(self):
        self.args = Namespace()
        self.args.source_segment_size = 500
        self.args.target_language = "eng"
        self.args.model_size = "dummy"
        self.args.fp16 = False
        self.args.num_beams = 5
        self.args.fairseq_dir = None
        self.args.device = "cpu"
        self.args.max_new_tokens = 100
        self.args.max_len = 200
        self.add_extra_args()
        self.agent = self.create_agent()

    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T._emit_remaining_tokens')
    def test_finish_read(self, mock_emit_remaining_tokens):
        mock_emit_remaining_tokens.return_value = WriteAction("dummy", finished=True)
        self.agent.states.source_finished = True
        self.assertIsInstance(self.agent.policy(self.agent.states), WriteAction)
