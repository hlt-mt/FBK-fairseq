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

import os
import unittest

import numpy as np
import torch
from torch import Tensor, tensor

from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.data.occlusion_dataset_genderxai import OccludedSpeechToTextDatasetGenderXai
from examples.speech_to_text.data.speech_to_text_dataset_genderxai import SpeechToTextDatasetGenderXai
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from examples.speech_to_text.occlusion_explanation.perturbators.discrete_fbank import \
    ContinuousOcclusionFbankPerturbator
from examples.speech_to_text.occlusion_explanation.perturbators.slic_fbank import \
    SlicOcclusionFbankPerturbatorFixedSegments
from fairseq.data import Dictionary, ConcatDataset
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset


class MockDictionary(Dictionary):
    def __init__(self):
        super().__init__()
        _ = self.add_symbol("_lui")
        _ = self.add_symbol("_and")
        _ = self.add_symbol("ava")
        _ = self.add_symbol("_al")
        _ = self.add_symbol("_parco")
        _ = self.add_symbol("_lei")


class MockDataConfig:
    @property
    def use_audio_input(self):
        return False

    @property
    def is_input_waveform(self):
        return False

    @property
    def shuffle(self):
        return False

    @property
    def waveform_sample_rate(self):
        return 16000

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        return False

    @property
    def input_feat_per_channel(self):
        return 80

    def get_feature_transforms(self, split, is_train):
        return None

    def get_waveform_transforms(self, split, is_train):
        return None


class TestOcclusionDatasetGenderXAI(unittest.TestCase):
    def setUp(self):
        self.mock_perturbator = ContinuousOcclusionFbankPerturbator(mask_probability=0.5, n_masks=5)
        current_directory = os.path.dirname(__file__)
        relative_path1 = os.path.join('mock_fbanks', 'array1.npy')
        relative_path2 = os.path.join('mock_fbanks', 'array2.npy')
        absolute_path1 = os.path.join(current_directory, relative_path1)
        absolute_path2 = os.path.join(current_directory, relative_path2)
        self.mock_dataset = SpeechToTextDatasetGenderXai(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=[absolute_path1, absolute_path2],
            n_frames=[100, 150],
            src_texts=["mock_src_text_1", "mock_src_text_2"],
            tgt_texts=["_lui mock_tgt_text", "_and _lei"],
            speakers=["speaker_1", "speaker_2"],
            src_langs=["it", "it"],
            tgt_langs=["it", "it"],
            ids=["id_1", "id_2"],
            tgt_dict=MockDictionary(),
            src_dict=MockDictionary(),
            pre_tokenizer=None,
            bpe_tokenizer=None,
            found_terms=["lui", "lei"],
            found_term_pairs=["lui lei", "lui lei"],
            gender_terms_indices=["0-0", "1-1"],
            swapped_tgt_texts=["_lei mock_tgt_text", "_and _lui"])
        self.concat_datasets = ConcatDataset([self.mock_dataset])
        self.occlusion_dataset = OccludedSpeechToTextDatasetGenderXai(
            to_be_occluded_dataset=self.concat_datasets,
            perturbator=self.mock_perturbator,
            tgt_dict=MockDictionary())
        
    def test_getitem(self):
        perturb_index, orig_dataset_index, mask, perturbed_fbank, \
            tgt_tokens, source_text, found_term, \
            found_term_pair, gender_term_index, swapped_tgt_tokens = self.occlusion_dataset[1]
        self.assertEqual(perturb_index, 1)
        self.assertEqual(orig_dataset_index, 0)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.size(), (100, 80))
        self.assertIsInstance(perturbed_fbank, Tensor)
        self.assertEqual(perturbed_fbank.size(), (100, 80))
        self.assertEqual(source_text.size(), (2,))
        self.assertEqual(tgt_tokens.tolist(), [4, 3, 2])  # lui <unk>, </s>
        self.assertEqual(found_term, "lui")
        self.assertEqual(found_term_pair, "lui lei")
        self.assertEqual(gender_term_index, "0-0")
        self.assertEqual(swapped_tgt_tokens.tolist(), [9, 3, 2])  # lei <unk>, </s>

    def test_collater_empty_samples(self):
        samples = []
        collated_data = self.occlusion_dataset.collater(samples)
        self.assertIsInstance(collated_data, dict)
        self.assertEqual(collated_data, {})

    def test_collater(self):
        samples = [
            (1, 0, torch.ones(100, 80), torch.randint(0, 1, (100, 80)), torch.tensor([5, 6, 2]), None,
             "lui", "lui lei", "0-0", torch.tensor([5, 9, 2])),
            (9, 1, torch.ones(120, 80), torch.randint(0, 1, (120, 80)), torch.tensor([6, 7, 8, 2]), None,
             "lei", "lui lei", "1-1", torch.tensor([9, 7, 8, 2]))]
        collated_data = self.occlusion_dataset.collater(samples)
        self.assertIsInstance(collated_data, dict)
        net_input = collated_data["net_input"]
        self.assertEqual(collated_data["nsentences"], 2)
        self.assertEqual(collated_data["id"].tolist(), [9, 1])
        self.assertEqual(collated_data["orig_id"].tolist(), [1, 0])
        self.assertEqual(collated_data["masks"].size(), (2, 120, 80))
        self.assertEqual(net_input["src_tokens"].size(), (2, 120, 80))
        self.assertEqual(net_input["src_lengths"].tolist(), [120, 100])
        self.assertTrue(torch.equal(net_input["prev_output_tokens"], tensor([[2, 6, 7, 8], [2, 5, 6, 1]])))
        self.assertTrue(torch.equal(collated_data["target"], tensor([[6, 7, 8, 2], [5, 6, 2, 1]])))
        self.assertEqual(collated_data["target_lengths"].tolist(), [4, 3])
        self.assertEqual(collated_data["src_texts"], [None, None])
        self.assertEqual(collated_data["found_terms"], ["lei", "lui"])
        self.assertEqual(collated_data["found_term_pairs"], ["lui lei", "lui lei"])
        self.assertEqual(collated_data["gender_terms_indices"], ["1-1", "0-0"])
        self.assertTrue(torch.equal(collated_data["swapped_target"], tensor([[9, 7, 8, 2], [5, 9, 2, 1]])))
        self.assertEqual(collated_data["swapped_target_lengths"].tolist(), [4, 3])


if __name__ == '__main__':
    unittest.main()
