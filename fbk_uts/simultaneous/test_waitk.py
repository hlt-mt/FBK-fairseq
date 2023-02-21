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
import copy

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import BOW_PREFIX
from examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk import WaitkAgent

from fbk_uts.simultaneous.test_base_simulst_agent import BaseSTAgentTestCase


class WaitkSimulSTPolicyTestCase(BaseSTAgentTestCase, unittest.TestCase):
    def add_extra_args(self):
        self.args.waitk = 0

    def create_agent(self):
        return WaitkAgent(self.args)

    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk.WaitkAgent.load_model_vocab')
    @patch('examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent.FairseqSimulSTAgent.__init__')
    def setUp(self, mock_load_model_vocab, mock_simulst_agent_init):
        mock_simulst_agent_init.return_value = None
        mock_load_model_vocab.return_value = None
        self.base_init()
        self.hypo = BOW_PREFIX + "I " + BOW_PREFIX + "am " + BOW_PREFIX + "a " + BOW_PREFIX + "quokka " + "."
        self.encoded_hypo = self.agent.tgtdict.encode_line(self.hypo, add_if_not_exist=False)
        self.states.n_audio_words = 4

    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk.WaitkAgent.new_hypo')
    def test_wait_0(self, mock_new_hypo):
        # Full hypothesis emitted
        mock_new_hypo.return_value = self.encoded_hypo
        self.states.n_predicted_words = 0
        WaitkAgent.waitk_prediction(self.agent, self.states)
        self.assertEqual(self.states.write, [4, 5, 6, 7, 8, 2])

    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk.WaitkAgent.new_hypo')
    def test_wait_2(self, mock_new_hypo):
        # Partial hypothesis emitted
        mock_new_hypo.return_value = self.encoded_hypo
        self.states.n_predicted_words = 0
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 2
        WaitkAgent.waitk_prediction(new_agent, self.states)
        self.assertEqual(self.states.write, [4, 5])

    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk.WaitkAgent.new_hypo')
    def test_wait_2_predicted_1(self, mock_new_hypo):
        # Partial hypothesis emitted considering already predicted words
        mock_new_hypo.return_value = self.encoded_hypo
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 2
        self.states.n_predicted_words = 1
        WaitkAgent.waitk_prediction(new_agent, self.states)
        self.assertEqual(self.states.write, [4])

    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk.WaitkAgent.new_hypo')
    def test_wait_4(self, mock_new_hypo):
        # No hypothesis emitted
        mock_new_hypo.return_value = self.encoded_hypo
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 4
        self.states.n_predicted_words = 0
        WaitkAgent.waitk_prediction(new_agent, self.states)
        self.assertEqual(self.states.write, [])

    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk.'
           'WaitkAgent._emit_remaining_tokens')
    def test_finish_read(self, mock_emit_remaining_tokens):
        mock_emit_remaining_tokens.return_values = None
        super().test_finish_read()


if __name__ == '__main__':
    unittest.main()
