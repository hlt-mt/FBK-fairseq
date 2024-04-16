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
from typing import Tuple

from torch import Tensor


LOGGER = logging.getLogger(__name__)


class Normalizer(ABC):
    """
    Normalizer classes take explanations and perform various types of normalizations,
    defined in the subclasses.
    """
    def __call__(
            self, fbank_explanation: Tensor, tgt_explanation: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


NORMALIZATION_REGISTRY = {}
NORMALIZATION_CLASS_NAMES = set()


def register_normalizer(name):
    def register_normalization_cls(cls):
        if name in NORMALIZATION_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate aggregator ({name})")
        if not issubclass(cls, Normalizer):
            raise ValueError(
                f"Aggregator ({name}: {cls.__name__}) must extend Aggregator")
        if cls.__name__ in NORMALIZATION_CLASS_NAMES:
            raise ValueError(
                f"Cannot register aggregator with duplicate class name ({cls.__name__})")
        NORMALIZATION_REGISTRY[name] = cls
        NORMALIZATION_CLASS_NAMES.add(cls.__name__)
        LOGGER.debug(f"Normalization registered: {name}.")
        return cls
    return register_normalization_cls


def get_normalizer(name):
    return NORMALIZATION_REGISTRY[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module(
            'examples.speech_to_text.occlusion_explanation.normalizers.' + module_name)
