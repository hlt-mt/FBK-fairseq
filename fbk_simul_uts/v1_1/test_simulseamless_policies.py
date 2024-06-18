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

from simuleval.agents import ReadAction, WriteAction

import unittest
from unittest.mock import patch

import torch

from examples.speech_to_text.simultaneous_translation.agents.speech_utils import BOW_PREFIX
from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_alignatt_seamlessm4t import AlignAttSeamlessS2T
from examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_waitk_seamlessm4t import WaitkSeamlessS2T
from fbk_simul_uts.v1_1.test_simulseamless_base import BaseSimulSeamlessTest


class WaitkSimulSeamlessTest(BaseSimulSeamlessTest, unittest.TestCase):
    def add_extra_args(self):
        self.args.waitk_lagging = 2
        self.args.continuous_write = 1

    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.load_model')
    def create_agent(self, mock_load_model):
        mock_load_model.return_value = None
        return WaitkSeamlessS2T(self.args)

    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.get_input_features')
    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.get_prefix')
    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.generate_and_decode')
    def test_wait2(self, mock_generate_and_decode, mock_get_prefix, mock_get_input_features):
        mock_get_input_features.return_value = None
        mock_get_prefix.return_value = None
        mock_generate_and_decode.return_value = "I am an example."
        self.agent.states.source_sample_rate = 16000

        # Wait 2, processed 1 -- read
        self.agent.states.source = torch.rand(8000)
        self.assertIsInstance(self.agent.policy(self.agent.states), ReadAction)

        # Wait 2, processed 2 -- write 1
        self.agent.states.source = torch.rand(16000)
        action = self.agent.policy(self.agent.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "I")

        # Wait 2, processed 2, numbers of words to emit 2 -- write 2
        self.agent.continuous_write = 2
        action = self.agent.policy(self.agent.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "I am")


class MockedTokenizer:
    def decode(self, hypo):
        return " ".join(["dummy"] * len(hypo))

    def convert_ids_to_tokens(self, hypo):
        return [BOW_PREFIX + "dummy"] * (len(hypo) + 1)


class AlignAttSimulSeamlessTest(BaseSimulSeamlessTest, unittest.TestCase):
    def add_extra_args(self):
        self.args.frame_num = 2
        self.args.extract_attn_from_layer = 1
        self.args.average_attn_by_layer = False

    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.load_model')
    def create_agent(self, mock_load_model):
        mock_load_model.return_value = None
        return AlignAttSeamlessS2T(self.args)

    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.get_input_features')
    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_base_seamlessm4t.'
        'SimulSeamlessS2T.get_prefix')
    @patch(
        'examples.speech_to_text.simultaneous_translation.agents.v1_1.'
        'simul_alignatt_seamlessm4t.AlignAttSeamlessS2T.generate_with_cross_attn')
    def test_alignatt(self, mock_generate_with_cross_attn, mock_get_prefix, mock_get_input_features):
        mock_get_input_features.return_value = None, None
        mock_get_prefix.return_value = []
        self.agent.tokenizer = MockedTokenizer()

        # Simulate 4 words predicted -- the third attends to forbidden frames
        mock_generate_with_cross_attn.return_value = [0, 1, 2, 3], torch.tensor([
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
                [0.03, 0.0, 0.05, 0.6, 0.02, 0.3],
                [0.2, 0.05, 0.05, 0.05, 0.35, 0.3],
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
            ])
        action = self.agent.policy(self.agent.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(len(action.content.split()), 2)

        # The third does not attend to a forbidden frame anymore, all output is emitted
        mock_generate_with_cross_attn.return_value = [0, 1, 2, 3], torch.tensor([
            [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
            [0.03, 0.0, 0.05, 0.6, 0.02, 0.3],
            [0.2, 0.35, 0.05, 0.05, 0.05, 0.3],
            [0.05, 0.35, 0.05, 0.5, 0.05, 0.05],
        ])
        action = self.agent.policy(self.agent.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(len(action.content.split()), 4)

        # The first attends to a forbidden frame, no output is emitted
        mock_generate_with_cross_attn.return_value = [0, 1, 2, 3], torch.tensor([
            [0.05, 0.05, 0.05, 0.05, 0.5, 0.3],
            [0.03, 0.0, 0.05, 0.6, 0.02, 0.3],
            [0.2, 0.35, 0.05, 0.05, 0.05, 0.3],
            [0.05, 0.35, 0.05, 0.5, 0.05, 0.05],
        ])
        action = self.agent.policy(self.agent.states)
        self.assertIsInstance(action, ReadAction)

    def test_get_words(self):
        # All tokens, the output contains all complete words (until "and", excluded)
        tokens = ["Hi", ",", BOW_PREFIX+"I", BOW_PREFIX+"am", BOW_PREFIX+"S", "ara", BOW_PREFIX+"and"]
        self.assertEqual(self.agent.get_words(tokens), "Hi, I am Sara")

        # All tokens but "and" as input, the output is emitted until "am", which is the last
        # complete word
        self.assertEqual(self.agent.get_words(tokens[:-1]), "Hi, I am")

        # First 2 tokens, output is None since no complete word is found (no sentencepiece BOS)
        self.assertEqual(self.agent.get_words(tokens[:2]), None)

        # Empty tokens list, output is None
        self.assertEqual(self.agent.get_words([]), None)


if __name__ == '__main__':
    unittest.main()
