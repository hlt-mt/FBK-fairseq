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
import logging
import os
from abc import ABC
from typing import Dict, Tuple, Union, List

from torch import Tensor


LOGGER = logging.getLogger(__name__)


class Aggregator(ABC):
    """
    Aggregator classes take explanations and perform various types of aggregations.
    The aggregation level, or the granularity of the aggregation result, varies for
    each subclass.
    """
    def __call__(
            self, explanations: Dict[int, Dict[str, Union[Tensor, str, List[str]]]]
    ) -> Dict[int, Dict[str, Union[Tensor, List[str]]]]:
        """
        Args:
            - explanations: dictionary containing a subdictionary for each sample_id.
            Each subdictionary contains "fbank_heatmap" tensor, "tgt_embed_heatmap" Tensor,
            "tgt_texts" list of strings, and possibly also "src_texts" string.
        Returns:
            - aggregated_explanations: same data structure of `explanations`.
        """
        raise NotImplementedError


AGGREGATION_REGISTRY = {}
AGGREGATION_CLASS_NAMES = set()


def register_aggregator(name):
    def register_aggregation_cls(cls):
        if name in AGGREGATION_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate aggregator ({name})")
        if not issubclass(cls, Aggregator):
            raise ValueError(
                f"Aggregator ({name}: {cls.__name__}) must extend Aggregator")
        if cls.__name__ in AGGREGATION_CLASS_NAMES:
            raise ValueError(
                f"Cannot register aggregator with duplicate class name ({cls.__name__})")
        AGGREGATION_REGISTRY[name] = cls
        AGGREGATION_CLASS_NAMES.add(cls.__name__)
        LOGGER.info(f"Aggregation registered: {name}.")
        return cls
    return register_aggregation_cls


def get_aggregator(name):
    return AGGREGATION_REGISTRY[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module(
            'examples.speech_to_text.occlusion_explanation.aggregators.' + module_name)
