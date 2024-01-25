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
import unittest
from unittest.mock import patch

import torch

from examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.streaming_st_agent import StreamingSTAgent, \
    get_class_from_string
from examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.text_first_history_selection import \
    PunctuationHistorySelection, FixedAudioHistorySelection
from simuleval.agents import ReadAction, WriteAction

from fbk_simul_uts.v1_1.test_base_simulst_agent import BaseSTAgentTestCaseV2, MockedLoadModelVocab


class MokedStreamingSTAgent:
    @staticmethod
    def fake_get_simulst_agent_from_args(args):
        simulst_agent_class = get_class_from_string(getattr(args, "simulst_agent_class"))
        return simulst_agent_class


class StreamAttSTPolicyTestCase(BaseSTAgentTestCaseV2, unittest.TestCase):
    def add_extra_args(self):
        self.args.simulst_agent_class = (
            "examples.speech_to_text.simultaneous_translation.agents.v1_1"
            ".simul_offline_alignatt.AlignAttSTAgent")
        self.args.history_selection_method = (
            "examples.speech_to_text.simultaneous_translation.agents.v1_1"
            ".streaming.text_first_history_selection.FixedWordsHistorySelection")
        self.args.attn_threshold = 0.0
        self.args.frame_num = 2
        self.args.extract_attn_from_layer = 1
        self.args.history_words = 2
        self.args.history_max_len = 100

    def create_agent(self):
        return StreamingSTAgent(self.args)

    def base_init(self):
        super().base_init()
        self.states = self.agent.simulst_agent.states
        self.states.encoder_states = "Dummy"
        self.states.new_segment = True

    def get_full_audio(self):
        # Fake encoded audio
        self.states.source = [torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])]

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent.load_model_vocab',
           MockedLoadModelVocab.load_model_vocab)
    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'base_simulst_agent.FairseqSimulSTAgent.to_device')
    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'speech_utils.OnlineFeatureExtractorV1_1.__call__')
    def setUp(self, mock_feature, mock_to_device):
        mock_to_device.return_value = None
        mock_feature.return_value = None
        self.base_init()
        self.get_full_audio()

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent._get_hypo_and_prefix')
    def test_no_prefix(self, get_hypo_and_prefix):
        # Standard AlignAtt policy with no prefix handling
        hypo = {
            "tokens": torch.tensor([4, 5, 6, 7, 0]),    # I am a quokka
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
                [0.6, 0.0, 0.05, 0.03, 0.02, 0.3],
                [0.2, 0.05, 0.05, 0.05, 0.35, 0.3],
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
            ]).transpose(0, 1)
        }
        # Prefix len 0
        get_hypo_and_prefix.return_value = hypo, 0
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "I")
        # Prefix len 1
        get_hypo_and_prefix.return_value = hypo, 1
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, ReadAction)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent._get_hypo_and_prefix')
    def test_prefix_fixed_word_selection(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 5, 6, 7, 0]),  # I am a quokka
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],   # first frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],    # second frame mostly attended
                [0.05, 0.5, 0.05, 0.05, 0.05, 0.3],   # second frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],    # second frame mostly attended
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],   # last frame mostly attended
            ]).transpose(0, 1)
        }

        # Case 1, prefix len 0
        # Since history_words is 2, we retain only "am a" from the new hypothesis,
        # and the first frame (only attended by "I") is discarded
        get_hypo_and_prefix.return_value = hypo, 0
        self.states.target_indices = []
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "I am a")
        # Check first 4 frames (corresponding to the first feature) discarded
        self.assertEqual(len(self.states.source[0]), 20
                         )

        # Case 2, prefix len 2: "I am"
        # Since history_words is 2, we retain only "am" from the prefix and "a"
        # from the new hypothesis, and the first frame (only attended by "I") is discarded
        get_hypo_and_prefix.return_value = hypo, 2
        # Restore fake encoded audio
        self.get_full_audio()
        self.states.target_indices = [4, 5]
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "a")
        self.assertEqual(len(self.states.source[0]), 20)

        # Case 3, prefix len 1: "I"
        hypo["attention"] = torch.tensor([
            [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],  # first frame mostly attended
            [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],   # second frame mostly attended
            [0.0, 0.05, 0.6, 0.03, 0.02, 0.3],   # third frame mostly attended
            [0.05, 0.05, 0.05, 0.05, 0.5, 0.3],  # last frame mostly attended
            [0.05, 0.05, 0.05, 0.05, 0.5, 0.3],  # last frame mostly attended
        ]).transpose(0, 1)
        get_hypo_and_prefix.return_value = hypo, 1
        # Restore fake encoded audio
        self.get_full_audio()
        self.states.target_indices = [4]
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "am")
        # Check no frames discarded
        self.assertEqual(len(self.states.source[0]), 24)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent._get_hypo_and_prefix')
    def test_prefix_punctuation_selection(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 8, 5, 7, 0]),  # I. am quokka
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],   # first frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],    # second frame mostly attended
                [0.05, 0.5, 0.05, 0.05, 0.05, 0.3],   # second frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],    # second frame mostly attended
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],   # last frame mostly attended
            ]).transpose(0, 1)
        }

        self.agent.history_selection_method = PunctuationHistorySelection(
            self.agent.simulst_agent.tgtdict, self.agent.simulst_agent.args)

        # Case 1, prefix len 1: "I"
        # A punctuation mark is present in the textual history (prefix "I" + new
        # hypothesis ". am"). The prefix and first new hypothesis token "." are
        # discarded together with the first feature. The second feature is retained
        # since it is attended by the second token "am" of the new hypothesis
        get_hypo_and_prefix.return_value = hypo, 1
        self.states.target_indices = [4]
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, ". am")
        # Check first 4 frames (corresponding to the first feature) discarded
        self.assertEqual(len(self.states.source[0]), 20)

        # Case 2, prefix len 2: "I."
        # A punctuation mark is present at the end of the prefix that is part of the
        # textual history. Therefore, the full prefix "I." is discarded together
        # with the first feature. The second feature is retained since it is attended
        # by the hypothesis "am"
        get_hypo_and_prefix.return_value = hypo, 2
        self.agent.states.target_indices = [4, 8]
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "am")
        self.assertEqual(len(self.states.source[0]), 20)

        # Case 3, prefix len 1: "I am"
        # No punctuation mark is present in the textual history, part of the
        # prefix "." is discarded while only "I" remains. Accordingly, the first
        # feature is discarded while the second feature is preserved since it is
        # attended by the selected prefix "I"
        hypo["tokens"] = torch.tensor([4, 5, 6, 7, 0])  # I am a quokka
        hypo["attention"] = torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],     # first frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],      # second frame mostly attended
                [0.05, 0.5, 0.05, 0.05, 0.05, 0.3],     # second frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],      # second frame mostly attended
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],     # last frame mostly attended
            ]).transpose(0, 1)
        get_hypo_and_prefix.return_value = hypo, 2
        self.agent.states.target_indices = [4, 5]
        # Restore fake encoded audio
        self.get_full_audio()
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "a")
        # Check first no frame discarded
        self.assertEqual(len(self.states.source[0]), 24)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent._get_hypo_and_prefix')
    def test_fixed_audio_selection(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 5, 7, 8, 0]),  # I am quokka.
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],  # first frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],  # second frame mostly attended
                [0.05, 0.5, 0.05, 0.05, 0.05, 0.3],  # second frame mostly attended
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],  # second frame mostly attended
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],  # last frame mostly attended
            ]).transpose(0, 1)
        }

        self.args.history_words = 1
        self.agent.history_selection_method = FixedAudioHistorySelection(
            self.agent.simulst_agent.tgtdict, self.agent.simulst_agent.args)

        # No prefix
        get_hypo_and_prefix.return_value = hypo, 0
        self.states.target_indices = []
        self.states.source = [torch.rand(280 // 10 * 4)]
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "I am")
        # "I am" should be written but only "am" should be retained as textual history (since
        # history_words is set to 1), therefore 280ms (corresponding to one word) should be
        # discarded
        self.assertEqual(len(self.states.source[0]), 280 // 10 * 3)

        # History len 1: "I"
        get_hypo_and_prefix.return_value = hypo, 1
        self.states.target_indices = [4]
        self.states.source = [torch.rand(280 // 10 * 4)]
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, WriteAction)
        self.assertEqual(action.content, "am")
        # "am" should be written and retained as textual history (since history_words is set to 1)
        # while "I" should be discarded, therefore 280ms (corresponding to one word) should be
        # discarded
        self.assertEqual(len(self.states.source[0]), 280 // 10 * 3)

        # History len 1: "am"
        get_hypo_and_prefix.return_value = hypo, 2
        self.agent.states.target_indices = [5]
        self.states.source = [torch.rand(280 // 10 * 4)]
        action = self.agent.policy(self.states)
        self.assertIsInstance(action, ReadAction)
        # Check no frame discarded
        self.assertEqual(len(self.states.source[0]), 280 // 10 * 4)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent._get_hypo_and_prefix')
    def test_no_token_emitted(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 5, 6, 7, 0]),
            "attention": torch.tensor([
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
                [0.03, 0.0, 0.05, 0.6, 0.02, 0.3],
                [0.2, 0.05, 0.05, 0.05, 0.35, 0.3],
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
                [0.05, 0.05, 0.05, 0.5, 0.05, 0.3],
            ]).transpose(0, 1)}
        # prefix len 0
        get_hypo_and_prefix.return_value = hypo, 0
        self.assertIsInstance(self.agent.policy(self.states), ReadAction)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent._get_hypo_and_prefix')
    def test_all_tokens_emitted(self, get_hypo_and_prefix):
        hypo = {
            "tokens": torch.tensor([4, 5, 6, 7, 0]),
            "attention": torch.tensor([
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
                [0.0, 0.6, 0.05, 0.03, 0.02, 0.3],
                [0.4, 0.05, 0.05, 0.05, 0.15, 0.3],
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
                [0.5, 0.05, 0.05, 0.05, 0.05, 0.3],
            ]).transpose(0, 1)}
        full_output_tokens = ["I", "am", "a", "quokka"]
        for prefix_len in range(3):
            get_hypo_and_prefix.return_value = hypo, prefix_len
            action = self.agent.policy(self.states)
            self.assertIsInstance(action, WriteAction)
            self.assertEqual(action.content, " ".join(full_output_tokens[prefix_len:-1]))
        get_hypo_and_prefix.return_value = hypo, 4
        self.assertIsInstance(self.agent.policy(self.states), ReadAction)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_1.'
           'simul_offline_alignatt.AlignAttSTAgent._emit_remaining_tokens')
    def test_finish_read(self, mock_emit_remaining_tokens):
        mock_emit_remaining_tokens.return_values = None
        super().test_finish_read()


if __name__ == '__main__':
    unittest.main()
