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
from argparse import Namespace

import torch
from simuleval.agents import ReadAction, WriteAction

from examples.speech_to_text.simultaneous_translation.agents.v1_1.base_simulst_agent import BOW_PREFIX, \
    FairseqSimulSTAgent
from examples.speech_to_text.simultaneous_translation.agents.v1_1.speech_utils import SpeechStates, \
    OnlineFeatureExtractor
from fairseq.data import Dictionary


class MockedLoadModelVocab:
    def load_model_vocab(self):
        self.model = torch.rand((1, 1))


class BaseSTAgentTestCaseV2:
    """
    It implements basic common tests to be executed for each v1_0
    ST agent as well as common parameter initialization. This class has to
    be inherited by the specific ST agent and two methods must be implemented:
    - *create_agent* in which the specific ST agent is instantiated
    - *add_extra_args* in which the parameters specific for the ST agent are
    added.
    """
    def add_extra_args(self):
        pass

    def create_agent(self):
        pass

    def base_init(self):
        self.args = Namespace()
        self.args.model_path = "dummy"
        self.args.data_bin = "dummy"
        self.args.shift_size = 10
        self.args.window_size = 25
        self.args.sample_rate = 16000
        self.args.feature_dim = 80
        self.args.global_cmvn = None
        self.args.config = None
        self.args.global_stats = None
        self.args.max_len = 200
        self.add_extra_args()
        self.agent = self.create_agent()
        self.initialize_agent(self.agent, self.args)
        self.states = SpeechStates(self.agent)
        self.agent.initialize_states(self.states)

    @staticmethod
    def initialize_agent(agent, args):
        agent.feature_extractor = OnlineFeatureExtractor(args)
        agent.eos = "<s>"
        agent.eos_idx = 0
        agent.prefix_token_idx = None
        agent.tgtdict = Dictionary()
        agent.tgtdict.add_symbol(BOW_PREFIX + "I")
        agent.tgtdict.add_symbol(BOW_PREFIX + "am")
        agent.tgtdict.add_symbol(BOW_PREFIX + "a")
        agent.tgtdict.add_symbol(BOW_PREFIX + "quokka")
        agent.tgtdict.add_symbol(".")

    def test_empty_encoder_states(self):
        self.states.encoder_states = None
        self.assertIsInstance(FairseqSimulSTAgent.policy(self.agent, self.states), ReadAction)

    def test_finish_read(self):
        self.states.encoder_states = "Dummy"
        self.states.source_finished = True
        self.assertIsInstance(FairseqSimulSTAgent.policy(self.agent, self.states), WriteAction)

    def test_no_new_input(self):
        self.states.new_segment = None
        self.assertIsInstance(FairseqSimulSTAgent.policy(self.agent, self.states), ReadAction)
