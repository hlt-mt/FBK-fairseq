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
import importlib
from argparse import ArgumentParser
import logging
from typing import Optional, List

from simuleval.agents import SpeechToTextAgent, AgentStates, Action
from simuleval.data.segments import Segment
from simuleval.utils.arguments import cli_argument_list

from examples.speech_to_text.simultaneous_translation.agents.base_simulst_agent import BaseSimulSTAgent
from examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.history_selection import HistorySelection


logger = logging.getLogger(__name__)


def get_class_from_string(class_name: str):
    try:
        agent_module = importlib.import_module(".".join(class_name.split(".")[:-1]))
        agent_class = getattr(agent_module, class_name.split(".")[-1])
    except Exception as e:
        logger.error(f"Not able to load {class_name}. Try setting --user-dir?")
        raise e
    return agent_class


def get_simulst_agent_from_args(args):
    simulst_agent_class = get_class_from_string(getattr(args, "simulst_agent_class"))
    if not issubclass(simulst_agent_class, BaseSimulSTAgent):
        raise Exception(
            "The selected agent class does not extend a BaseSimulSTAgent class. Please select a "
            "proper SimulST agent class.")
    return simulst_agent_class


def get_history_selection_from_args(args):
    history_selection_class = get_class_from_string(getattr(args, "history_selection_method"))
    if not issubclass(history_selection_class, HistorySelection):
        raise Exception(
            "The selected agent class does not extend a HistorySelection class. Please select a "
            "proper history selection class.")
    return history_selection_class


class StreamingSTAgent(SpeechToTextAgent):
    """
    Base agent for Streaming Speech Translation policies which works with SimulEval>=1.1.0.
    It includes a generic method *policy* that implements the logic of the streaming policy and
    that requires:
      - *simulst_agent.policy*: the SimulST agent policy that decides, given the audio and text
      history, whether the predicted hypothesis can be paritally or completely emitted;
      - *history_selection_method*: the logic used to select the past audio and textual information
       to retain from the history that will be used in the next step prediction.
    """
    def __init__(self, args):
        simulst_agent_cls = get_simulst_agent_from_args(args)
        self.simulst_agent = simulst_agent_cls(args)
        history_selection_method_cls = get_history_selection_from_args(args)
        self.history_selection_method = history_selection_method_cls(self.simulst_agent.tgtdict, args)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--simulst-agent-class", type=str, required=True,
            help="Name of the simul ST agent class to use.")
        parser.add_argument(
            "--history-selection-method", type=str, required=True,
            help="Name of the class to use to determine the text and audio "
                 "suffixes to keep as context for next predictions")
        args, _ = parser.parse_known_args(cli_argument_list(None))
        simulst_agent_cls = get_simulst_agent_from_args(args)
        history_selection_method_cls = get_history_selection_from_args(args)
        simulst_agent_cls.add_args(parser)
        history_selection_method_cls.add_args(parser)

    def build_states(self) -> AgentStates:
        return self.simulst_agent.build_states()

    def reset(self) -> None:
        self.simulst_agent.reset()

    def policy(self, states: Optional[AgentStates] = None) -> Action:
        # Execute SimulST policy that determines whether and what to write
        action = self.simulst_agent.policy(states)
        # Clear useless past audio and text history
        self.history_selection_method(action, states)
        return action

    def push(
            self,
            source_segment: Segment,
            states: Optional[AgentStates] = None,
            upstream_states: Optional[List[AgentStates]] = None) -> None:
        super().push(source_segment, self.simulst_agent.states, upstream_states)

    def pop(self, states: Optional[AgentStates] = None) -> Segment:
        return super().pop(self.simulst_agent.states)

    def to(self, device: str, *args, **kwargs) -> None:
        self.simulst_agent.to(device, args, kwargs)
