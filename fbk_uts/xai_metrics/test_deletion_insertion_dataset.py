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
import tempfile
import unittest

import numpy as np
import torch
from torch import Tensor, tensor

from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from examples.speech_to_text.xai_metrics.deletion_insertion_dataset import \
    FeatureAttributionEvaluationSpeechToTextDataset
from fairseq.data import Dictionary, ConcatDataset
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset


class MockDictionary(Dictionary):
    def __init__(self):
        super().__init__()
        for symbol in ["_lui", "_and", "ava", "_al", "_parco"]:
            self.add_symbol(symbol)


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
        return 5

    def get_feature_transforms(self, split, is_train):
        return None

    def get_waveform_transforms(self, split, is_train):
        return None


class TestMetricSpeechToTextDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary files for mock fbank
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f1, \
                tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f2:
            # Generate random data for mock files
            fbank_0_data = np.array(
                [[12, 15, 21, 0, 3], [27, 3, 7, 9, 19], [21, 18, 4, 23, 6], [24, 24, 12, 26, 1]])
            fbank_1_data = np.array(
                [[6, 7, 23, 14, 24],
                 [17, 5, 25, 13, 8],
                 [9, 20, 19, 16, 19],
                 [5, 15, 15, 0, 18],
                 [3, 24, 17, 19, 29],
                 [19, 19, 14, 7, 0]])
            np.save(f1, fbank_0_data)
            np.save(f2, fbank_1_data)
            absolute_path1 = f1.name
            absolute_path2 = f2.name
        self.temp_files = [absolute_path1, absolute_path2]

        self.aggregated_fbank_heatmaps = {
            0: {"fbank_heatmap": torch.tensor(
                [[20, 20, 2, 1, 5],
                 [16, 14, 12, 1, 0],
                 [2, 3, 1, 10, 12],
                 [12, 12, 12, 12, 20]], dtype=torch.float32),
                "tgt_embed_heatmap": torch.tensor([], dtype=torch.float32),
                "tgt_text": []},
            1: {"fbank_heatmap": torch.tensor(
                [[20, 20, 2, 1, 5],  # 9
                 [16, 14, 12, 1, 0],  # 14
                 [2, 3, 1, 10, 12],  # 19
                 [12, 12, 12, 12, 20],  # 24
                 [2, 2, 1, 4, 0],  # 29
                 [5, 1, 4, 6, 0]], dtype=torch.float32),  # 34
                "tgt_embed_heatmap": torch.tensor([], dtype=torch.float32),
                "tgt_text": []}}

        self.mock_dataset = SpeechToTextDataset(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=[absolute_path1, absolute_path2],
            n_frames=[4, 6],
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

        self.metric_dataset_deletion = FeatureAttributionEvaluationSpeechToTextDataset(
            original_dataset=self.concat_datasets,
            aggregated_fbank_heatmaps=self.aggregated_fbank_heatmaps,
            interval_size=5,
            num_intervals=21,
            metric="deletion")

        self.metric_dataset_insertion = FeatureAttributionEvaluationSpeechToTextDataset(
            original_dataset=self.concat_datasets,
            aggregated_fbank_heatmaps=self.aggregated_fbank_heatmaps,
            interval_size=10,
            num_intervals=11,
            metric="insertion")

        self.mock_dataset_with_src = SpeechToTextDatasetWithSrc(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=[absolute_path1, absolute_path2],
            n_frames=[4, 6],
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

        self.metric_dataset_with_src_deletion = FeatureAttributionEvaluationSpeechToTextDataset(
            original_dataset=self.concat_datasets_with_src,
            aggregated_fbank_heatmaps=self.aggregated_fbank_heatmaps,
            interval_size=10,
            num_intervals=11,
            metric="deletion")

        self.metric_dataset_with_src_insertion = FeatureAttributionEvaluationSpeechToTextDataset(
            original_dataset=self.concat_datasets_with_src,
            aggregated_fbank_heatmaps=self.aggregated_fbank_heatmaps,
            interval_size=10,
            num_intervals=11,
            metric="insertion")

    def tearDown(self):
        # Clean up the temporary files
        for file_path in self.temp_files:
            os.remove(file_path)

    def test_original_index(self):
        perturb_indices = [4, 28, 65]
        mapped_indices = [
            self.metric_dataset_deletion._original_index(perturb_index) for perturb_index in perturb_indices]
        self.assertEqual(mapped_indices, [0, 1, 3])

    def test_len(self):
        actual_len = len(self.metric_dataset_deletion)
        self.assertEqual(actual_len, 42)

    # check that for dataset_with_src the size of returned tuple contains also src_text
    def test_getitem_size(self):
        returned_tuple = self.metric_dataset_deletion[1]
        returned_tuple_with_src = self.metric_dataset_with_src_deletion[1]
        self.assertEqual(len(returned_tuple), 3)
        self.assertEqual(len(returned_tuple_with_src), 4)

    # insertion
    def test_get_item(self):
        perturb_index, perturbed_fbank, tgt_text = self.metric_dataset_insertion[6]
        expected_fbank = tensor(
            [[12., 15., 0., 0., 0.],
             [27., 3., 7., 0., 0.],
             [0., 0., 0., 23., 6.],
             [24., 24., 12., 26., 1.]])
        self.assertEqual(perturb_index, 6)
        self.assertIsInstance(perturbed_fbank, Tensor)
        self.assertEqual(perturbed_fbank.size(), (4, 5))
        self.assertEqual(tgt_text.tolist(), [4, 3, 2])  # '_lui': 4, '<unk>': 3, '</s>': 2
        self.assertTrue(torch.equal(perturbed_fbank, expected_fbank))

    # deletion
    def test_getitem_with_src(self):
        self.metric_dataset_with_src_deletion.id_to_ordered_indices = {}  # reset attribute
        perturb_index, perturbed_fbank, tgt_text, src_text = self.metric_dataset_with_src_deletion[18]
        expected_fbank = tensor(
            [[0., 0., 0., 14., 0.],
             [0., 0., 0., 13., 8.],
             [0., 0., 19., 0., 0.],
             [0., 0., 0., 0., 0.],
             [3., 0., 17., 0., 29.],
             [0., 19., 0., 0., 0.]])
        self.assertEqual(perturb_index, 18)
        self.assertIsInstance(perturbed_fbank, Tensor)
        self.assertEqual(perturbed_fbank.size(), (6, 5))
        self.assertEqual(src_text.tolist(), [3, 2])  # '<unk>': 3, '</s>': 2
        self.assertEqual(tgt_text.tolist(), [5, 2])  # '_and': 5, '</s>': 2
        self.assertTrue(torch.equal(perturbed_fbank, expected_fbank))

    # check that for the first percentage value in deletion, nothing is zeroed out
    def test_getitem_deletion_first(self):
        self.metric_dataset_with_src_deletion.id_to_ordered_indices = {}  # reset attribute
        perturb_index, perturbed_fbank, tgt_text, src_text = self.metric_dataset_with_src_deletion[11]
        self.assertTrue(perturbed_fbank.sum() == 429)

    # check that for the last percentage value in deletion, everything is zeroed out
    def test_getitem_deletion_last(self):
        self.metric_dataset_deletion.id_to_ordered_indices = {}  # reset attribute
        perturb_index, perturbed_fbank, tgt_text = self.metric_dataset_deletion[20]
        self.assertTrue(perturbed_fbank.sum() == 0)

    # check that for the first percentage value in insertion, everything is zeroed out
    def test_getitem_insertion_first(self):
        self.metric_dataset_insertion.id_to_ordered_indices = {}  # reset attribute
        perturb_index, perturbed_fbank, tgt_text = self.metric_dataset_insertion[11]
        self.assertTrue(perturbed_fbank.sum() == 0)

    # check that for the last percentage value in insertion, nothing is zeroed out
    def test_getitem_insertion_last(self):
        self.metric_dataset_with_src_insertion.id_to_ordered_indices = {}  # reset attribute
        perturb_index, perturbed_fbank, tgt_text, src_text = self.metric_dataset_with_src_insertion[10]
        self.assertTrue(perturbed_fbank.sum() == 275)

    def test_collater_empty_samples(self):
        samples = []
        collated_data = self.metric_dataset_deletion.collater(samples)
        self.assertIsInstance(collated_data, dict)
        self.assertEqual(collated_data, {})

    def test_collater(self):
        samples = [
            (27, torch.randint(0, 30, (4, 5)), torch.tensor([5, 2])),
            (50, torch.randint(0, 30, (6, 5)), torch.tensor([3, 3, 2]))]
        collated_data = self.metric_dataset_deletion.collater(samples)
        self.assertIsInstance(collated_data, dict)

        self.assertEqual(collated_data["nsentences"], 2)
        self.assertEqual(collated_data["id"].tolist(), [50, 27])
        self.assertEqual(collated_data["orig_id"].tolist(), [2, 1])

        net_input = collated_data["net_input"]
        self.assertEqual(net_input["src_tokens"].size(), (2, 6, 5))
        self.assertEqual(net_input["src_lengths"].tolist(), [6, 4])
        self.assertTrue(torch.equal(net_input["prev_output_tokens"], torch.tensor([[2, 3, 3], [2, 5, 1]])))

        self.assertTrue(torch.equal(collated_data["target"], tensor([[3, 3, 2], [5, 2, 1]])))
        self.assertEqual(collated_data["target_lengths"].tolist(), [3, 2])
        self.assertEqual(collated_data["ntokens"], 5)

    def test_collater_with_src(self):
        samples = [
            (7, torch.randint(0, 30, (4, 5)), torch.tensor([5, 2]), torch.tensor([3, 2])),
            (37, torch.randint(0, 30, (6, 5)), torch.tensor([3, 3, 2]), torch.tensor([3, 5, 2]))]
        collated_data = self.metric_dataset_with_src_deletion.collater(samples)
        self.assertIsInstance(collated_data, dict)

        self.assertEqual(collated_data["nsentences"], 2)
        self.assertEqual(collated_data["id"].tolist(), [37, 7])
        self.assertEqual(collated_data["orig_id"].tolist(), [3, 0])

        net_input = collated_data["net_input"]
        self.assertEqual(net_input["src_tokens"].size(), (2, 6, 5))
        self.assertEqual(net_input["src_lengths"].tolist(), [6, 4])
        self.assertTrue(torch.equal(net_input["prev_output_tokens"], torch.tensor([[2, 3, 3], [2, 5, 1]])))
        self.assertTrue(torch.equal(net_input["prev_transcript_tokens"], torch.tensor([[2, 3, 5], [2, 3, 1]])))

        self.assertTrue(torch.equal(collated_data["target"], tensor([[3, 3, 2], [5, 2, 1]])))
        self.assertEqual(collated_data["target_lengths"].tolist(), [3, 2])
        self.assertTrue(torch.equal(collated_data["transcript"], tensor([[3, 5, 2], [3, 2, 1]])))
        self.assertEqual(collated_data["transcript_lengths"].tolist(), [3, 2])
        self.assertEqual(collated_data["ntokens"], 5)
        self.assertEqual(collated_data["ntokens_transcript"], 5)

    def test_sizes(self):
        expected_sizes = [
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        new_sizes = self.metric_dataset_deletion.sizes
        self.assertEqual(new_sizes.tolist(), expected_sizes)

    def test_num_tokens(self):
        num_tokens_id12 = self.metric_dataset_with_src_deletion.num_tokens(12)
        num_tokens_id9 = self.metric_dataset_with_src_deletion.num_tokens(9)
        self.assertEqual(num_tokens_id12, 6)
        self.assertEqual(num_tokens_id9, 4)

    def test_size(self):
        size_id22 = self.metric_dataset_deletion.size(22)
        size_id9 = self.metric_dataset_deletion.size(9)
        self.assertEqual(size_id22, (6, 1))
        self.assertEqual(size_id9, (4, 2))

    def test_ordered_indices(self):
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
        aggregated_fbank_heatmaps = {
            1: {"fbank_heatmap": torch.rand(2, 3), "tgt_embed_heatmap": torch.rand(5, 2), "tgt_text": []},
            2: {"fbank_heatmap": torch.rand(2, 3), "tgt_embed_heatmap": torch.rand(5, 2), "tgt_text": []},
            3: {"fbank_heatmap": torch.rand(2, 3), "tgt_embed_heatmap": torch.rand(5, 2), "tgt_text": []},
            4: {"fbank_heatmap": torch.rand(2, 3), "tgt_embed_heatmap": torch.rand(5, 2), "tgt_text": []}}
        metric_dataset = FeatureAttributionEvaluationSpeechToTextDataset(
            original_dataset=concat_datasets,
            aggregated_fbank_heatmaps=aggregated_fbank_heatmaps,
            interval_size=10,
            num_intervals=11,
            metric="deletion")
        expected_ordered_indices = metric_dataset.ordered_indices()
        self.assertEqual(
            expected_ordered_indices.tolist(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43])

    def test_attr(self):
        shuffle_config_id0 = self.metric_dataset_deletion.attr("shuffle", 0)
        shuffle_config_id7 = self.metric_dataset_deletion.attr("shuffle", 7)
        self.assertEqual(shuffle_config_id0, False)
        self.assertEqual(shuffle_config_id7, False)


if __name__ == '__main__':
    unittest.main()
