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

import os
import unittest

import torch
from torch import Tensor, tensor

from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from examples.speech_to_text.occlusion_explanation.encoder_perturbator import OcclusionFbankPerturbatorContinuous, \
    SlicOcclusionFbankPerturbator
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


class TestOcclusionDataset(unittest.TestCase):
    def setUp(self):
        self.mock_perturbator = OcclusionFbankPerturbatorContinuous(mask_probability=0.5, n_masks=5)
        current_directory = os.path.dirname(__file__)
        relative_path1 = os.path.join('mock_fbanks', 'array1.npy')
        relative_path2 = os.path.join('mock_fbanks', 'array2.npy')
        absolute_path1 = os.path.join(current_directory, relative_path1)
        absolute_path2 = os.path.join(current_directory, relative_path2)
        self.mock_dataset = SpeechToTextDataset(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=[absolute_path1, absolute_path2],
            n_frames=[100, 150],
            src_texts=None,
            tgt_texts=["_lui mock_tgt_text", "_and"],
            speakers=["speaker_1", "speaker_2"],
            src_langs=["it", "it"],
            tgt_langs=["it", "it"],
            ids=["id_1", "id_2"],
            tgt_dict=MockDictionary(),
            pre_tokenizer=None,
            bpe_tokenizer=None)
        self.concat_datasets = ConcatDataset([self.mock_dataset])
        self.occlusion_dataset = OccludedSpeechToTextDataset(
            to_be_occluded_dataset=self.concat_datasets,
            perturbator=self.mock_perturbator,
            tgt_dict=MockDictionary())
        self.mock_dataset_with_src = SpeechToTextDatasetWithSrc(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=[absolute_path1, absolute_path2],
            n_frames=[100, 150],
            src_texts=["mock_src_text_1", "mock_src_text_2"],
            tgt_texts=["_lui mock_tgt_text", "_and"],
            speakers=["speaker_1", "speaker_2"],
            src_langs=["it", "it"],
            tgt_langs=["it", "it"],
            ids=["id_1", "id_2"],
            tgt_dict=MockDictionary(),
            src_dict=MockDictionary(),
            pre_tokenizer=None,
            bpe_tokenizer=None,
            bpe_tokenizer_src=None)
        self.concat_datasets_with_src = ConcatDataset([self.mock_dataset_with_src])
        self.occlusion_dataset_with_src = OccludedSpeechToTextDataset(
            to_be_occluded_dataset=self.concat_datasets_with_src,
            perturbator=self.mock_perturbator,
            tgt_dict=MockDictionary())

    # Check that n_masks is dynamically assigned based on the type of perturbator
    def test_attribute_n_masks(self):
        mock_slic_perturbator = SlicOcclusionFbankPerturbator(
            mask_probability=0.5, n_masks=5, segments_range=(2, 3), segments_step=1, slic_sigma=5)
        slic_occlusion_dataset = OccludedSpeechToTextDataset(
            to_be_occluded_dataset=self.concat_datasets,
            perturbator=mock_slic_perturbator,
            tgt_dict=MockDictionary())
        self.assertEqual(slic_occlusion_dataset.n_masks, 4)
        self.assertTrue(self.occlusion_dataset.n_masks, 5)

    def test_original_index(self):
        perturb_indices = [7, 6, 0]
        mapped_indices = [
            self.occlusion_dataset._original_index(perturb_index) for perturb_index in perturb_indices]
        self.assertEqual(mapped_indices, [1, 1, 0])

    def test_len(self):
        actual_len = len(self.occlusion_dataset)
        self.assertEqual(actual_len, 10)

    def test_getitem(self):
        perturb_index, orig_dataset_index, mask, perturbed_fbank, \
            predicted_tokens, source_text = self.occlusion_dataset[1]
        self.assertEqual(perturb_index, 1)
        self.assertEqual(orig_dataset_index, 0)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.size(), (100, 80))
        self.assertIsInstance(perturbed_fbank, Tensor)
        self.assertEqual(perturbed_fbank.size(), (100, 80))
        self.assertEqual(source_text, None)
        self.assertEqual(predicted_tokens.tolist(), [4, 3, 2])  # lui <unk>, </s>

    def test_getitem_with_src(self):
        perturb_index, orig_dataset_index, mask, perturbed_fbank, \
            predicted_tokens, source_text = self.occlusion_dataset_with_src[6]
        self.assertEqual(perturb_index, 6)
        self.assertEqual(orig_dataset_index, 1)
        self.assertIsInstance(mask, Tensor)
        self.assertEqual(mask.size(), (150, 80))
        self.assertIsInstance(perturbed_fbank, Tensor)
        self.assertEqual(perturbed_fbank.size(), (150, 80))
        self.assertEqual(source_text.tolist(), [3, 2])  # <unk> </s>
        self.assertEqual(predicted_tokens.tolist(), [5, 2])

    def test_collater_empty_samples(self):
        samples = []
        collated_data = self.occlusion_dataset.collater(samples)
        self.assertIsInstance(collated_data, dict)
        self.assertEqual(collated_data, {})

    def test_collater(self):
        samples = [
            (1, 0, torch.ones(100, 80), torch.randint(0, 1, (100, 80)), torch.tensor([5, 6, 2]), None),
            (9, 1, torch.ones(120, 80), torch.randint(0, 1, (120, 80)), torch.tensor([6, 7, 8, 2]), None)]
        collated_data = self.occlusion_dataset.collater(samples)
        self.assertIsInstance(collated_data, dict)
        net_input = collated_data["net_input"]
        self.assertEqual(collated_data["nsentences"], 2)
        self.assertEqual(collated_data["id"].tolist(), [9, 1])
        self.assertEqual(collated_data["orig_id"].tolist(), [1, 0])
        self.assertEqual(collated_data["masks"].size(), (2, 120, 80))
        self.assertEqual(net_input["src_tokens"].size(), (2, 120, 80))
        self.assertEqual(net_input["src_lengths"].tolist(), [120, 100])
        self.assertTrue(torch.equal(net_input["target"], tensor([[2, 6, 7, 8], [2, 5, 6, 1]])))
        self.assertEqual(net_input["target_lengths"].tolist(), [4, 3])
        self.assertEqual(collated_data["src_texts"], [None, None])

    def test_collater_with_src(self):
        samples = [
            (4, 0, torch.ones(150, 80), torch.randint(0, 1, (150, 80)),
             torch.tensor([5, 5, 2]), torch.tensor([4, 3, 2])),
            (6, 1, torch.ones(110, 80), torch.randint(0, 1, (110, 80)),
             torch.tensor([6, 2]), torch.tensor([5, 2])),
            (1, 0, torch.ones(100, 80), torch.randint(0, 1, (100, 80)),
             torch.tensor([6, 5, 2]), torch.tensor([5, 4, 2]))]
        collated_data = self.occlusion_dataset_with_src.collater(samples)
        net_input = collated_data["net_input"]
        self.assertEqual(collated_data["nsentences"], 3)
        self.assertEqual(collated_data["id"].tolist(), [4, 6, 1])
        self.assertEqual(collated_data["orig_id"].tolist(), [0, 1, 0])
        self.assertEqual(collated_data["masks"].size(), (3, 150, 80))
        self.assertEqual(net_input["src_tokens"].size(), (3, 150, 80))
        self.assertEqual(net_input["src_lengths"].tolist(), [150, 110, 100])
        self.assertTrue(torch.equal(net_input["target"], tensor([[2, 5, 5], [2, 6, 1], [2, 6, 5]])))
        self.assertEqual(net_input["target_lengths"].tolist(), [3, 2, 3])
        self.assertTrue(torch.equal(collated_data["src_texts"][0], torch.tensor([4, 3, 2])))
        self.assertTrue(torch.equal(collated_data["src_texts"][1], torch.tensor([5, 2])))
        self.assertTrue(torch.equal(collated_data["src_texts"][2], torch.tensor([5, 4, 2])))

    def test_sizes(self):
        expected_sizes = [100, 100, 100, 100, 100, 150, 150, 150, 150, 150]
        new_sizes = self.occlusion_dataset.sizes
        self.assertEqual(new_sizes.tolist(), expected_sizes)

    def test_num_tokens(self):
        num_tokens_id2 = self.occlusion_dataset_with_src.num_tokens(2)
        num_tokens_id7 = self.occlusion_dataset_with_src.num_tokens(7)
        self.assertEqual(num_tokens_id2, 100)
        self.assertEqual(num_tokens_id7, 150)

    def test_size(self):
        size_id2 = self.occlusion_dataset.size(2)
        size_id7 = self.occlusion_dataset.size(7)
        self.assertEqual(size_id2, (100, 2))
        self.assertEqual(size_id7, (150, 1))

    def test_ordered_indices(self):
        mock_perturbator = OcclusionFbankPerturbatorContinuous(mask_probability=0.5, n_masks=4)
        mock_dataset = SpeechToTextDataset(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=["path", "path", "path", "path"],
            n_frames=[100, 150, 122, 179],
            src_texts=None,
            tgt_texts=None,
            speakers=None,
            src_langs=None,
            tgt_langs=None,
            ids=None,
            tgt_dict=None,
            pre_tokenizer=None,
            bpe_tokenizer=None)
        concat_datasets = ConcatDataset([mock_dataset])
        occlusion_dataset = OccludedSpeechToTextDataset(
            to_be_occluded_dataset=concat_datasets,
            perturbator=mock_perturbator,
            tgt_dict=MockDictionary())
        expected_ordered_indices = occlusion_dataset.ordered_indices()
        self.assertEqual(expected_ordered_indices.tolist(), [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15])

    def test_attr(self):
        shuffle_config_id0 = self.occlusion_dataset.attr("shuffle", 0)
        shuffle_config_id7 = self.occlusion_dataset.attr("shuffle", 7)
        self.assertEqual(shuffle_config_id0, False)
        self.assertEqual(shuffle_config_id7, False)


if __name__ == '__main__':
    unittest.main()
