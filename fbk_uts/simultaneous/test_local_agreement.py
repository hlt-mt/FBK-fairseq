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

from examples.speech_to_text.simultaneous_translation.agents.simul_offline_local_agreement import \
    LocalAgreementSimulSTAgent

from simuleval.states import SpeechStates

from examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk import OnlineFeatureExtractor


class LocalAgreementSimulSTPolicyTestCase(unittest.TestCase):

    @classmethod
    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_local_agreement.LocalAgreementSimulSTAgent.load_model_vocab')
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
        self.initialize_agent(self)
        self.initialize_state(self)

    @patch('examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk.FairseqSimulSTAgent.__init__')
    def initialize_agent(self, mock_agent_init):
        mock_agent_init.return_value = None
        self.agent = LocalAgreementSimulSTAgent(self.args)
        self.agent.feature_extractor = OnlineFeatureExtractor(self.args)
        self.agent.eos = "<s>"
        self.agent.eos_idx = 0

    def initialize_state(self):
        self.states = SpeechStates(None, None, 0, self.agent)
        self.agent.initialize_states(self.states)

    def test_incomplete_prefix(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."], ["I", "am", "an", "elephant."]]
        prefix = LocalAgreementSimulSTAgent.prefix(self.agent, self.states)
        self.assertEqual(prefix, ["I", "am"])

    def test_complete_prefix(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."], ["I", "am", "a", "quokka."]]
        prefix = LocalAgreementSimulSTAgent.prefix(self.agent, self.states)
        self.assertEqual(prefix, ["I", "am", "a", "quokka."])

    def test_empty_prefix(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."], ["Hello", "I", "am", "a", "quokka."]]
        prefix = LocalAgreementSimulSTAgent.prefix(self.agent, self.states)
        self.assertEqual(prefix, [])

    def test_empty_chunks(self):
        self.states.chunks_hyp = []
        prefix = LocalAgreementSimulSTAgent.prefix(self.agent, self.states)
        self.assertEqual(prefix, [])

    def test_one_chunk(self):
        self.states.chunks_hyp = [["I", "am", "a", "quokka."]]
        prefix = LocalAgreementSimulSTAgent.prefix(self.agent, self.states)
        self.assertEqual(prefix, [])

    def test_three_chunks(self):
        self.states.chunks_hyp = [
            ["Hello", "I", "am", "a", "quokka."],
            ["I", "am", "a", "quokka."],
            ["I", "am", "an", "elephant."]
        ]
        prefix = LocalAgreementSimulSTAgent.prefix(self.agent, self.states)
        self.assertEqual(prefix, ["I", "am"])
