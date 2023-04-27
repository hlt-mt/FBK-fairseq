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
import unittest
from unittest.mock import patch

import torch
from simuleval.agents import ReadAction, WriteAction

from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_edatt import EDAttSTAgent
from fbk_simul_uts.v1_1.test_base_simulst_agent import BaseSTAgentTestCaseV2, MockedLoadModelVocab


class EDAttSimulSTPolicyV2TestCase(BaseSTAgentTestCaseV2, unittest.TestCase):
    def add_extra_args(self):
        self.args.attn_threshold = 0.2
        self.args.frame_num = 2
        self.args.extract_attn_from_layer = 1

    def create_agent(self):
        return EDAttSTAgent(self.args)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_edatt.EDAttSTAgent.load_model_vocab', MockedLoadModelVocab.load_model_vocab)
    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'base_simulst_agent.FairseqSimulSTAgent.to_device')
    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'speech_utils.OnlineFeatureExtractorV1_1.__call__')
    def setUp(self, mock_feature, mock_to_device):
        mock_to_device.return_value = None
        mock_feature.return_value = None
        self.base_init()

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_edatt.EDAttSTAgent._get_hypo_and_prefix')
    def test_first_token_below_threshold(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 5, 6, 7]),
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
                [0.6, 0.0, 0.05, 0.03, 0.02, 0.3],
                [0.2, 0.35, 0.05, 0.05, 0.05, 0.3],
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
            ]).transpose(0, 1)
        }
        # prefix len 0
        get_hypo_and_prefix.return_value = hypo, 0
        action = EDAttSTAgent._policy(self.agent, self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "I")
        # prefix len 1
        get_hypo_and_prefix.return_value = hypo, 1
        self.assertIsInstance(EDAttSTAgent._policy(self.agent, self.states), ReadAction)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1'
           '.simul_offline_edatt.EDAttSTAgent._get_hypo_and_prefix')
    def test_no_tokens_below_threshold(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 5, 6, 7]),
            "attention": torch.tensor([
                [0.05, 0.05, 0.05, 0.05, 0.5, 0.3],
                [0.02, 0.0, 0.05, 0.03, 0.6, 0.3],
                [0.05, 0.35, 0.05, 0.05, 0.2, 0.3],
                [0.05, 0.05, 0.05, 0.05, 0.5, 0.3],
            ]).transpose(0, 1)
        }
        get_hypo_and_prefix.return_value = hypo, 0
        self.assertIsInstance(EDAttSTAgent._policy(self.agent, self.states), ReadAction)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_edatt.EDAttSTAgent._get_hypo_and_prefix')
    def test_all_tokens_below_threshold(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 5, 6, 7]),
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
                [0.6, 0.0, 0.05, 0.03, 0.02, 0.3],
                [0.24, 0.35, 0.05, 0.05, 0.01, 0.3],
                [0.58, 0.05, 0.05, 0.01, 0.01, 0.3],
            ]).transpose(0, 1)
        }
        full_output_tokens = ["I", "am", "a", "quokka"]
        for prefix_len in range(3):
            get_hypo_and_prefix.return_value = hypo, prefix_len
            action = EDAttSTAgent._policy(self.agent, self.states)
            self.assertIsInstance(action, WriteAction)
            self.assertEqual(action.content, " ".join(full_output_tokens[prefix_len:-1]))
        get_hypo_and_prefix.return_value = hypo, 4
        self.assertIsInstance(EDAttSTAgent._policy(self.agent, self.states), ReadAction)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_edatt.EDAttSTAgent._emit_remaining_tokens')
    def test_finish_read(self, mock_emit_remaining_tokens):
        mock_emit_remaining_tokens.return_values = None
        super().test_finish_read()


if __name__ == '__main__':
    unittest.main()
