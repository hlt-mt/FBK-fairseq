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
import os
import logging
import h5py
from typing import Dict, Tuple
import torch

from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel
from examples.speech_to_text.occlusion_explanation.perturbators import OcclusionFbankPerturbator
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.dictionary import Dictionary


LOGGER = logging.getLogger(__name__)


class ExplanationTask:
    """
    Base class for occlusion explanation tasks. It contains the methods that change 
    based on the explanation task at hand (e.g. explaining the model's prediction for an entire
    sentence, or only for gender terms).
    """

    @staticmethod
    def get_occluded_dataset(
            to_be_occluded_dataset: ConcatDataset,
            perturbator: OcclusionFbankPerturbator,
            tgt_dict: Dictionary) -> OccludedSpeechToTextDataset:
        """
        Args:
            - to_be_occluded_dataset: the dataset to be occluded.
            - perturbator: the perturbator to be used to occlude the data.
            - tgt_dict: the target dictionary.
        Returns:
            An OccludedSpeechToTextDataset object, its specific class depends on the explanation 
            task at hand.
        """
        raise NotImplementedError
    
    @staticmethod
    def save_original_probs(model: S2TTransformerModel, sample: Dict, save_file: str) -> None:
        """
        Performs a forward pass of the model on the sample and saves the output probabilities in a h5 file.
        Args:
            - model: the model to be used to compute the probabilities.
            - sample: the utterances to be used as input to the model, it should
                        contain the tokenized target text on which to perform forced decoding.
            - save_file: the path to the h5 file where to save the probabilities.
        """
        raise NotImplementedError
    
    @staticmethod
    def read_original_probs(original_probs_file: h5py.File, sample: Dict) -> Dict:
        """
        Args:
            - original_probs_file: open h5py file containing the original probabilities ouput 
                                   by the model.
            - sample: dictionary containing the sample on which to compute explanations.
        Returns:
            - original_probs: a dictionary in the form {id: Tensor (padded_seq_len, dict_len)}
                              containing the original probabilities relevant for the 
                              task at hand for each utterance in the sample.
        """
        raise NotImplementedError
    
    @staticmethod
    def get_perturbed_probs(model: S2TTransformerModel, sample: Dict) -> Dict:
        """
        Args:
            - model: a model with a decoder of type OcclusionTransformerDecoderScriptable
                     to be used to compute the perturbed probabilities.
            - sample: dictionary containing the sample on which to compute explanations.
        Returns:
            A dictionary containing the perturbed probabilities relevant for the task at hand
            and the corresponding mask(s) indicating which target embeddings where occluded.
        """
        raise NotImplementedError


TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def register_explanation_task(name):
    def register_explanation_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate task ({name})")
        if not issubclass(cls, ExplanationTask):
            raise ValueError(
                f"Explanation task ({name}: {cls.__name__}) must extend ExplanationTask")
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                f"Cannot register explanation task with duplicate class name ({cls.__name__})")
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        LOGGER.debug(f"Explanation task registered: {name}.")
        return cls
    return register_explanation_task_cls


def get_explanation_task(name):
    return TASK_REGISTRY[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module(
            'examples.speech_to_text.occlusion_explanation.explanation_tasks.' + module_name)
