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

from typing import Dict, Tuple
import torch
import h5py
from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.data.occlusion_dataset_genderxai import OccludedSpeechToTextDatasetGenderXai
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel
from examples.speech_to_text.occlusion_explanation.explanation_tasks import ExplanationTask, register_explanation_task
from examples.speech_to_text.occlusion_explanation.perturbators import OcclusionFbankPerturbator
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.dictionary import Dictionary 


@register_explanation_task("gender")
class GenderExplanationTask(ExplanationTask):
    """
    The goal of this task is to explain the gender terms the model outputs. It aims to find
    the most relevant features that make the model generate a term in one gender instead of another.
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
            Since here the task is to explain the gender of gender terms, we return an 
            OccludedSpeechToTextDatasetGenderXai object, which contains special fields with 
            gender term annotations.
        """
        return OccludedSpeechToTextDatasetGenderXai(to_be_occluded_dataset, perturbator, tgt_dict)
