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

from argparse import Namespace
import torch

from examples.speech_to_text.simultaneous_translation.agents.simul_offline_edatt import EDAttSTAgent

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import OnlineFeatureExtractor
from fbk_uts.simultaneous.test_local_agreement import LocalAgreementSimulSTPolicyTestCase


class EDAttSimulSTPolicyTestCase(unittest.TestCase):

    @classmethod
    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_edatt.EDAttSTAgent.load_model_vocab')
    def setUpClass(self, mock_load_model_vocab):
        mock_load_model_vocab.return_value = None
        self.args = Namespace()
        self.args.model_path = "dummy"
        self.args.data_bin = "dummy"
        self.args.shift_size = 10
        self.args.window_size = 25
        self.args.sample_rate = 16000
        self.args.feature_dim = 80
        self.args.global_cmvn = None
        self.args.attn_threshold = 0.2
        self.args.frame_num = 2
        self.args.extract_attn_from_layer = 1
        self.initialize_agent(self)
        LocalAgreementSimulSTPolicyTestCase.initialize_state(self)

    @patch('examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent.FairseqSimulSTAgent.__init__')
    def initialize_agent(self, mock_agent_init):
        mock_agent_init.return_value = None
        self.agent = EDAttSTAgent(self.args)
        self.agent.feature_extractor = OnlineFeatureExtractor(self.args)
        self.agent.eos = "<s>"
        self.agent.eos_idx = 0
        self.agent.prefix_token_idx = 0

    def test_first_two_tokens_below_threshold(self):
        hypo = {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
                [0.6, 0.0, 0.05, 0.03, 0.02, 0.3],
                [0.2, 0.35, 0.05, 0.05, 0.05, 0.3],
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
            ]).transpose(0, 1)
        }
        prefix_len = 0
        self.assertTrue(EDAttSTAgent.check_attention_threshold(self.agent, hypo, prefix_len, self.states))
        self.assertTrue(
            torch.equal(self.states.write, torch.tensor([1, 2], dtype=self.states.write.dtype))
        )
        prefix_len = 1
        self.assertTrue(EDAttSTAgent.check_attention_threshold(self.agent, hypo, prefix_len, self.states))
        self.assertTrue(
            torch.equal(self.states.write, torch.tensor([2], dtype=self.states.write.dtype))
        )
        prefix_len = 2
        self.assertFalse(EDAttSTAgent.check_attention_threshold(self.agent, hypo, prefix_len, self.states))

    def test_no_tokens_below_threshold(self):
        hypo = {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "attention": torch.tensor([
                [0.05, 0.05, 0.05, 0.05, 0.5, 0.3],
                [0.02, 0.0, 0.05, 0.03, 0.6, 0.3],
                [0.05, 0.35, 0.05, 0.05, 0.2, 0.3],
                [0.05, 0.05, 0.05, 0.05, 0.5, 0.3],
            ]).transpose(0, 1)
        }
        prefix_len = 0
        self.assertFalse(EDAttSTAgent.check_attention_threshold(self.agent, hypo, prefix_len, self.states))

    def test_all_tokens_below_threshold(self):
        hypo = {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
                [0.6, 0.0, 0.05, 0.03, 0.02, 0.3],
                [0.24, 0.35, 0.05, 0.05, 0.01, 0.3],
                [0.58, 0.05, 0.05, 0.01, 0.01, 0.3],
            ]).transpose(0, 1)
        }
        full_output_tokens = [1, 2, 3, 4]
        for prefix_len in range(3):
            self.assertTrue(EDAttSTAgent.check_attention_threshold(self.agent, hypo, prefix_len, self.states))
            self.assertTrue(
                torch.equal(
                    self.states.write,
                    torch.tensor(full_output_tokens[prefix_len:], dtype=self.states.write.dtype)))
        prefix_len = 4
        self.assertFalse(EDAttSTAgent.check_attention_threshold(self.agent, hypo, prefix_len, self.states))


if __name__ == '__main__':
    unittest.main()
