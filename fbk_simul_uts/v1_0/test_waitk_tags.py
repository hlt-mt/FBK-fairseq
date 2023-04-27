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

import torch

from examples.speech_to_text.simultaneous_translation.agents.speech_utils import BOW_PREFIX
from examples.speech_to_text.simultaneous_translation.agents.v1_0.simul_offline_waitk_tags import WaitkAgentWithTags

from fbk_simul_uts.v1_0.test_base_simulst_agent import BaseSTAgentTestCase


class WaitkSimulSTWithTagsTestCase(BaseSTAgentTestCase, unittest.TestCase):
    def add_extra_args(self):
        self.args.waitk = 0
        self.args.parallel = False

    def create_agent(self):
        return WaitkAgentWithTags(self.args)

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_waitk_tags.WaitkAgentWithTags.load_model_vocab')
    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'base_simulst_agent.FairseqSimulSTAgent.__init__')
    def setUp(self, mock_load_model_vocab, mock_simulst_agent_init):
        mock_simulst_agent_init.return_value = None
        mock_load_model_vocab.return_value = None
        self.base_init()
        self.hypo = BOW_PREFIX + "quokka " + BOW_PREFIX + "is " + BOW_PREFIX + "pretty ."
        self.agent.tgtdict.add_symbol(BOW_PREFIX + "is")
        self.agent.tgtdict.add_symbol(BOW_PREFIX + "pretty")
        self.agent.tgtdict.add_symbol("<PERSON>")
        self.agent.tgtdict.add_symbol("</PERSON>")
        self.agent.tags = ["", "", "", "", "", "", "", "", "", "", "PERSON"]
        self.encoded_hypo = self.agent.tgtdict.encode_line(self.hypo, add_if_not_exist=False)
        self.predicted_tags = torch.tensor([self.agent.tgtdict.index("<PERSON>"), 0, 0, 0, 0])
        self.states.n_audio_words = 3

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_waitk_tags.WaitkAgentWithTags.new_hypo')
    def test_full_hypo(self, mock_new_hypo):
        # Full hypothesis emitted
        mock_new_hypo.return_value = self.encoded_hypo, self.predicted_tags
        self.states.n_predicted_words = 0
        WaitkAgentWithTags.waitk_prediction(self.agent, self.states)
        self.assertEqual(self.states.write, [11, 7, 12, 9, 10, 8, 2])

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_waitk_tags.WaitkAgentWithTags.new_hypo')
    def test_wait_1(self, mock_new_hypo):
        # Partial hypothesis emitted
        mock_new_hypo.return_value = self.encoded_hypo, self.predicted_tags
        self.states.n_predicted_words = 0
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 1
        WaitkAgentWithTags.waitk_prediction(new_agent, self.states)
        self.assertEqual(self.states.write, [11, 7, 12, 9])

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_waitk_tags.WaitkAgentWithTags.new_hypo')
    def test_wait_1_predicted_1(self, mock_new_hypo):
        # Partial hypothesis emitted considering already predicted words
        mock_new_hypo.return_value = self.encoded_hypo[1:], self.predicted_tags[1:]
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 1
        self.states.n_predicted_words = 1
        WaitkAgentWithTags.waitk_prediction(new_agent, self.states)
        self.assertEqual(self.states.write, [9])

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_waitk_tags.WaitkAgentWithTags.new_hypo')
    def test_wait_3(self, mock_new_hypo):
        # No hypothesis emitted
        mock_new_hypo.return_value = self.encoded_hypo, self.predicted_tags
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 3
        self.states.n_predicted_words = 0
        WaitkAgentWithTags.waitk_prediction(new_agent, self.states)
        self.assertEqual(self.states.write, [])

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_waitk_tags.WaitkAgentWithTags.new_hypo')
    def test_emit_remaining_tokens_with_tags(self, mock_new_hypo):
        mock_new_hypo.return_value = self.encoded_hypo, self.predicted_tags
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 3
        self.states.n_predicted_words = 0
        WaitkAgentWithTags._emit_remaining_tokens(new_agent, self.states)
        self.assertEqual(self.states.write, [11, 7, 12, 9, 10, 8, 2])

        # Move tag towards the end (last word: "pretty")
        mock_new_hypo.return_value = self.encoded_hypo, torch.tensor(
            [0, 0, self.agent.tgtdict.index("<PERSON>"), 0, 0])
        new_agent = copy.deepcopy(self.agent)
        new_agent.waitk = 3
        self.states.n_predicted_words = 0
        WaitkAgentWithTags._emit_remaining_tokens(new_agent, self.states)
        self.assertEqual(self.states.write, [7, 9, 11, 10, 12, 8, 2])

    @patch('examples.speech_to_text.simultaneous_translation.agents.v1_0.'
           'simul_offline_waitk_tags.WaitkAgentWithTags._emit_remaining_tokens')
    def test_finish_read(self, mock_emit_remaining_tokens):
        mock_emit_remaining_tokens.return_values = None
        super().test_finish_read()


if __name__ == '__main__':
    unittest.main()
