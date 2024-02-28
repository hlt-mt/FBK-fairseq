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

import argparse
import os
import unittest

import h5py
import torch

from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from examples.speech_to_text.occlusion_explanation.aggregator import Aggregator, decode_line
from examples.speech_to_text.occlusion_explanation.perturbators.discrete_fbank import ContinuousOcclusionFbankPerturbator
from fairseq.data import ConcatDataset
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fbk_uts.occlusion_explanation.test_occlusion_dataset import MockDataConfig, MockDictionary


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace()
        current_directory = os.path.dirname(__file__)
        self.args.save_file = current_directory + "/file"
        self.mock_dict = MockDictionary()
        self.mock_perturbator = ContinuousOcclusionFbankPerturbator(mask_probability=0.5, n_masks=2)
        self.mock_dataset = SpeechToTextDataset(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=["mock_path/to/audio1", "mock_path/to/audio2"],
            n_frames=[100, 150],
            src_texts=None,
            tgt_texts=["_lui mock_tgt_text", "_and"],
            speakers=["speaker_1", "speaker_2"],
            src_langs=["it", "it"],
            tgt_langs=["it", "it"],
            ids=["id_1", "id_2"],
            tgt_dict=self.mock_dict,
            pre_tokenizer=None,
            bpe_tokenizer=None)
        self.concat_datasets = ConcatDataset([self.mock_dataset])
        self.occlusion_dataset = OccludedSpeechToTextDataset(
            to_be_occluded_dataset=self.concat_datasets,
            perturbator=self.mock_perturbator,
            tgt_dict=self.mock_dict)
        self.aggregator = Aggregator(self.args, self.occlusion_dataset)
        self.collated_data_batch1 = {
            "orig_id": torch.tensor([0, 1, 0]),
            "src_texts": ["source text 1", "source text 2", "source text 1"],
            "net_input": {
                "src_lengths": torch.LongTensor([6, 7, 6]),
                "target": torch.tensor([[2, 4, 3, 1], [2, 3, 4, 6], [2, 4, 3, 1]]),
                "target_lengths": torch.LongTensor([3, 4, 3])}}
        self.collated_data_batch2 = {
            "orig_id": torch.tensor([1, 1, 2, 2]),
            "src_texts": ["source text 2", "source text 2", "source text 3", "source text 3"],
            "net_input": {
                "src_lengths": torch.LongTensor([7, 7, 9, 9]),
                "target": torch.tensor(
                    [[2, 3, 4, 6, 1], [2, 3, 4, 6, 1], [2, 6, 3, 4, 5], [2, 6, 3, 4, 5]]),
                "target_lengths": torch.LongTensor([4, 4, 5, 5])}}
        # the size of fbank_heatmaps is (batch size, sequence length, time, channels)
        self.fbank_heatmaps_batch1 = torch.ones(3, 4, 7, 10)
        self.fbank_heatmaps_batch2 = torch.ones(4, 5, 9, 10)
        self.fbank_masks_batch1 = torch.ones(3, 1, 7, 10)
        self.fbank_masks_batch2 = torch.ones(4, 1, 9, 10)
        # the size of tgt_embed_heatmaps is (batch size, sequence length, sequence length, embedding dim)
        self.tgt_embed_heatmaps_batch1 = torch.ones(3, 4, 4, 24)
        self.tgt_embed_heatmaps_batch2 = torch.ones(4, 5, 5, 24)
        self.tgt_embed_masks_batch1 = torch.ones(3, 1, 4, 24)
        self.tgt_embed_masks_batch2 = torch.ones(4, 1, 5, 24)

    def test_decode_line_without_tokenizers(self):
        result = decode_line(torch.LongTensor([2, 4, 3, 1]), self.mock_dict)
        self.assertEqual(result, ['</s>', '_lui', '<unk>'])

    # test decode_line() with a SpeechToTextDatasetWithSrc having a src_dict
    def test_decode_line_dataset_with_src(self):
        mock_dataset = SpeechToTextDatasetWithSrc(
            split="mock_split",
            is_train_split=False,
            data_cfg=MockDataConfig(),
            audio_paths=["mock_path/to/audio1", "mock_path/to/audio2"],
            n_frames=[100, 150],
            src_texts=["source text 1", "source text 2"],
            tgt_texts=["_lui mock_tgt_text", "_and"],
            speakers=["speaker_1", "speaker_2"],
            src_langs=["it", "it"],
            tgt_langs=["it", "it"],
            ids=["id_1", "id_2"],
            src_dict=self.mock_dict,
            tgt_dict=self.mock_dict,
            pre_tokenizer=None,
            bpe_tokenizer=None)
        concat_datasets = ConcatDataset([mock_dataset])
        occlusion_dataset = OccludedSpeechToTextDataset(
            to_be_occluded_dataset=concat_datasets,
            perturbator=self.mock_perturbator,
            tgt_dict=self.mock_dict)
        aggregator = Aggregator(self.args, occlusion_dataset)
        collated_data_batch = {
            "orig_id": torch.tensor([0, 1, 0]),
            "src_texts": torch.tensor([[2, 4, 3, 1], [2, 3, 4, 6], [2, 4, 3, 1]]),
            "net_input": {
                "src_lengths": torch.LongTensor([6, 7, 6]),
                "target": torch.tensor([[2, 4, 3, 1], [2, 3, 4, 6], [2, 4, 3, 1]]),
                "target_lengths": torch.LongTensor([3, 4, 3])}}
        aggregator._update_heatmaps(
            collated_data_batch,
            self.fbank_heatmaps_batch1,
            self.fbank_masks_batch1,
            self.tgt_embed_heatmaps_batch1,
            self.tgt_embed_masks_batch1)
        self.assertEqual(aggregator.final_masks[0]["src_text"], ['</s>', '_lui', '<unk>'])
        self.assertEqual(aggregator.final_masks[1]["src_text"], ['</s>', '<unk>', '_lui', 'ava'])

    # test _update_heatmaps() with a single batch involved
    def test_update_heatmaps_single_batch(self):
        previous_fbank_heatmap = torch.zeros(4, 7, 10)
        previous_tgt_embed_heatmap = torch.zeros(4, 4, 24)
        previous_fbank_mask = torch.zeros(1, 7, 10)
        previous_tgt_embed_mask = torch.zeros(1, 4, 24)
        self.aggregator.final_masks = {1: {
            "fbank_heatmap": previous_fbank_heatmap,
            "tgt_embed_heatmap": previous_tgt_embed_heatmap,
            "fbank_mask": previous_fbank_mask,
            "tgt_embed_mask": previous_tgt_embed_mask,
            "n_masks": 2}}
        self.aggregator._update_heatmaps(
            self.collated_data_batch1,
            self.fbank_heatmaps_batch1,
            self.fbank_masks_batch1,
            self.tgt_embed_heatmaps_batch1,
            self.tgt_embed_masks_batch1)
        # check if size of heatmaps is preserved
        self.assertEqual(self.aggregator.final_masks[0]["fbank_heatmap"].shape, (3, 6, 10))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_heatmap"].shape, (3, 3, 24))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_heatmap"].shape, (4, 7, 10))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_heatmap"].shape, (4, 4, 24))
        # check if size of masks is preserved
        self.assertEqual(self.aggregator.final_masks[0]["fbank_mask"].shape, (1, 6, 10))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_mask"].shape, (1, 3, 24))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_mask"].shape, (1, 7, 10))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_mask"].shape, (1, 4, 24))
        # check if existing entries are updated
        self.assertEqual(self.aggregator.final_masks[1]["n_masks"], 3)
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["fbank_heatmap"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["tgt_embed_heatmap"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["fbank_mask"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["tgt_embed_mask"] == 1))
        self.assertEqual(len(self.aggregator.final_masks), 2)
        self.assertEqual(self.aggregator.final_masks[0]["n_masks"], 2)
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["fbank_heatmap"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["tgt_embed_heatmap"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["fbank_mask"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["tgt_embed_mask"] == 2))
        self.assertEqual(self.aggregator.final_masks[0]["src_text"], "source text 1")

    # test _update_heatmaps() with continuous heatmaps for filterbanks and
    # discrete heatmaps for target embeddings (last dimension is 1)
    def test_update_heatmaps_discrete_tgt(self):
        previous_fbank_heatmap = torch.zeros(4, 7, 10)  # (seq_len, time, channels)
        previous_tgt_embed_heatmap = torch.zeros(4, 4, 1)
        previous_fbank_mask = torch.zeros(1, 7, 10)  # (1, time, channels)
        previous_tgt_embed_mask = torch.zeros(1, 4, 1)
        self.aggregator.final_masks = {1: {
            "fbank_heatmap": previous_fbank_heatmap,
            "tgt_embed_heatmap": previous_tgt_embed_heatmap,
            "fbank_mask": previous_fbank_mask,
            "tgt_embed_mask": previous_tgt_embed_mask,
            "n_masks": 2}}
        self.aggregator._update_heatmaps(
            self.collated_data_batch1,
            torch.ones(3, 4, 7, 10),
            torch.ones(3, 1, 7, 10),
            torch.ones(3, 4, 4, 1),
            torch.ones(3, 1, 4, 1))
        # check if size of heatmaps is preserved
        self.assertEqual(self.aggregator.final_masks[0]["fbank_heatmap"].shape, (3, 6, 10))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_heatmap"].shape, (3, 3, 1))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_heatmap"].shape, (4, 7, 10))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_heatmap"].shape, (4, 4, 1))
        # check if size of masks is preserved
        self.assertEqual(self.aggregator.final_masks[0]["fbank_mask"].shape, (1, 6, 10))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_mask"].shape, (1, 3, 1))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_mask"].shape, (1, 7, 10))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_mask"].shape, (1, 4, 1))

    # test _update_heatmaps() with discrete heatmaps for filterbanks (last dimension is 1)
    # and continuous heatmaps for target embeddings.
    def test_update_heatmaps_discrete_fbank(self):
        previous_fbank_heatmap = torch.zeros(4, 7, 1)  # (seq_len, time, 1)
        previous_tgt_embed_heatmap = torch.zeros(4, 4, 24)
        previous_fbank_mask = torch.zeros(1, 7, 1)  # (1, time, 1)
        previous_tgt_embed_mask = torch.zeros(1, 4, 24)
        self.aggregator.final_masks = {1: {
            "fbank_heatmap": previous_fbank_heatmap,
            "tgt_embed_heatmap": previous_tgt_embed_heatmap,
            "fbank_mask": previous_fbank_mask,
            "tgt_embed_mask": previous_tgt_embed_mask,
            "n_masks": 2}}
        self.aggregator._update_heatmaps(
            self.collated_data_batch1,
            torch.ones(3, 4, 7, 1),
            torch.ones(3, 1, 7, 1),
            torch.ones(3, 4, 4, 24),
            torch.ones(3, 1, 4, 24))
        # check if size of heatmaps is preserved
        self.assertEqual(self.aggregator.final_masks[0]["fbank_heatmap"].shape, (3, 6, 1))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_heatmap"].shape, (3, 3, 24))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_heatmap"].shape, (4, 7, 1))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_heatmap"].shape, (4, 4, 24))
        # check if size of masks is preserved
        self.assertEqual(self.aggregator.final_masks[0]["fbank_mask"].shape, (1, 6, 1))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_mask"].shape, (1, 3, 24))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_mask"].shape, (1, 7, 1))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_mask"].shape, (1, 4, 24))

    # test _update_heatmaps() with two batches involved and different padding lengths
    def test_update_heatmaps_multiple_batch(self):
        self.aggregator._update_heatmaps(
            self.collated_data_batch1,
            self.fbank_heatmaps_batch1,
            self.fbank_masks_batch1,
            self.tgt_embed_heatmaps_batch1,
            self.tgt_embed_masks_batch1)
        self.aggregator._update_heatmaps(
            self.collated_data_batch2,
            self.fbank_heatmaps_batch2,
            self.fbank_masks_batch2,
            self.tgt_embed_heatmaps_batch2,
            self.tgt_embed_masks_batch2)
        # check if size of heatmaps is preserved, with padding stripped correctly
        self.assertEqual(self.aggregator.final_masks[0]["fbank_heatmap"].shape, (3, 6, 10))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_heatmap"].shape, (3, 3, 24))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_heatmap"].shape, (4, 7, 10))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_heatmap"].shape, (4, 4, 24))
        self.assertEqual(self.aggregator.final_masks[2]["fbank_heatmap"].shape, (5, 9, 10))
        self.assertEqual(self.aggregator.final_masks[2]["tgt_embed_heatmap"].shape, (5, 5, 24))
        # check if size of masks is preserved, with padding stripped correctly
        self.assertEqual(self.aggregator.final_masks[0]["fbank_mask"].shape, (1, 6, 10))
        self.assertEqual(self.aggregator.final_masks[0]["tgt_embed_mask"].shape, (1, 3, 24))
        self.assertEqual(self.aggregator.final_masks[1]["fbank_mask"].shape, (1, 7, 10))
        self.assertEqual(self.aggregator.final_masks[1]["tgt_embed_mask"].shape, (1, 4, 24))
        self.assertEqual(self.aggregator.final_masks[2]["fbank_mask"].shape, (1, 9, 10))
        self.assertEqual(self.aggregator.final_masks[2]["tgt_embed_mask"].shape, (1, 5, 24))
        # check if n. masks are correctly updated
        self.assertEqual(self.aggregator.final_masks[0]["n_masks"], 2)
        self.assertEqual(self.aggregator.final_masks[1]["n_masks"], 3)
        self.assertEqual(self.aggregator.final_masks[2]["n_masks"], 2)
        # check if heatmaps and masks are correctly updated
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["fbank_heatmap"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["tgt_embed_heatmap"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["fbank_heatmap"] == 3))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["tgt_embed_heatmap"] == 3))
        self.assertTrue(torch.all(self.aggregator.final_masks[2]["fbank_heatmap"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[2]["tgt_embed_heatmap"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["fbank_mask"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["tgt_embed_mask"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["fbank_mask"] == 3))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["tgt_embed_mask"] == 3))
        self.assertTrue(torch.all(self.aggregator.final_masks[2]["fbank_mask"] == 2))
        self.assertTrue(torch.all(self.aggregator.final_masks[2]["tgt_embed_mask"] == 2))
        # check src_texts
        self.assertEqual(self.aggregator.final_masks[0]["src_text"], "source text 1")
        self.assertEqual(self.aggregator.final_masks[1]["src_text"], "source text 2")
        self.assertEqual(self.aggregator.final_masks[2]["src_text"], "source text 3")
        # check tgt_texts
        self.assertEqual(self.aggregator.final_masks[0]["tgt_text"], ['</s>', '_lui', '<unk>'])
        self.assertEqual(self.aggregator.final_masks[1]["tgt_text"], ['</s>', '<unk>', '_lui', 'ava'])
        self.assertEqual(self.aggregator.final_masks[2]["tgt_text"], ['</s>', 'ava', '<unk>', '_lui', '_and'])

    def test_normalize(self):
        self.aggregator._update_heatmaps(
            self.collated_data_batch1,
            self.fbank_heatmaps_batch1,
            self.fbank_masks_batch1,
            self.tgt_embed_heatmaps_batch1,
            self.tgt_embed_masks_batch1)
        self.aggregator._update_heatmaps(
            self.collated_data_batch2,
            self.fbank_heatmaps_batch2,
            self.fbank_masks_batch2,
            self.tgt_embed_heatmaps_batch2,
            self.tgt_embed_masks_batch2)
        self.aggregator._normalize(0)
        self.aggregator._normalize(1)
        self.aggregator._normalize(2)
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["fbank_heatmap"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[0]["tgt_embed_heatmap"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["fbank_heatmap"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[1]["tgt_embed_heatmap"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[2]["fbank_heatmap"] == 1))
        self.assertTrue(torch.all(self.aggregator.final_masks[2]["tgt_embed_heatmap"] == 1))

    def test_normalize_division_by_zero(self):
        self.aggregator.final_masks = {0: {}}
        self.aggregator.final_masks[0]["fbank_heatmap"] = torch.ones(3, 4, 5)
        self.aggregator.final_masks[0]["tgt_embed_heatmap"] = torch.ones(3, 3, 12)
        self.aggregator.final_masks[0]["fbank_mask"] = torch.tensor(
            [[[2, 1, 0, 1, 2],
              [2, 2, 0, 1, 1],
              [2, 0, 1, 0, 1],
              [2, 2, 2, 1, 1]]])
        self.aggregator.final_masks[0]["tgt_embed_mask"] = torch.tensor(
            [[[2, 1, 0, 1, 2, 1, 2, 2, 1, 2, 0, 0],
              [2, 2, 0, 1, 2, 1, 1, 1, 0, 0, 1, 1],
              [2, 0, 1, 0, 2, 2, 2, 0, 1, 1, 1, 1]]])
        expected_fbank_heatmap = torch.tensor(
            [[[0.5, 1, 0, 1, 0.5],
              [0.5, 0.5, 0, 1, 1],
              [0.5, 0, 1, 0, 1],
              [0.5, 0.5, 0.5, 1, 1]],
             [[0.5, 1, 0, 1, 0.5],
              [0.5, 0.5, 0, 1, 1],
              [0.5, 0, 1, 0, 1],
              [0.5, 0.5, 0.5, 1, 1]],
             [[0.5, 1, 0, 1, 0.5],
              [0.5, 0.5, 0, 1, 1],
              [0.5, 0, 1, 0, 1],
              [0.5, 0.5, 0.5, 1, 1]]])
        expected_tgt_heatmap = torch.tensor(
            [[[0.5, 1, 0, 1, 0.5, 1, 0.5, 0.5, 1, 0.5, 0, 0],
              [0.5, 0.5, 0, 1, 0.5, 1, 1, 1, 0, 0, 1, 1],
              [0.5, 0, 1, 0, 0.5, 0.5, 0.5, 0, 1, 1, 1, 1]],
             [[0.5, 1, 0, 1, 0.5, 1, 0.5, 0.5, 1, 0.5, 0, 0],
              [0.5, 0.5, 0, 1, 0.5, 1, 1, 1, 0, 0, 1, 1],
              [0.5, 0, 1, 0, 0.5, 0.5, 0.5, 0, 1, 1, 1, 1]],
             [[0.5, 1, 0, 1, 0.5, 1, 0.5, 0.5, 1, 0.5, 0, 0],
              [0.5, 0.5, 0, 1, 0.5, 1, 1, 1, 0, 0, 1, 1],
            [0.5, 0, 1, 0, 0.5, 0.5, 0.5, 0, 1, 1, 1, 1]]])
        self.aggregator._normalize(0)
        self.assertTrue(torch.equal(
            self.aggregator.final_masks[0]["fbank_heatmap"], expected_fbank_heatmap))
        self.assertTrue(torch.equal(
            self.aggregator.final_masks[0]["tgt_embed_heatmap"], expected_tgt_heatmap))

    def test_call_method(self):
        file_path = self.aggregator.save_file + ".h5"
        try:
            self.aggregator.__call__(
                self.collated_data_batch1,
                self.fbank_heatmaps_batch1,
                self.fbank_masks_batch1,
                self.tgt_embed_heatmaps_batch1,
                self.tgt_embed_masks_batch1)
            self.aggregator.__call__(
                self.collated_data_batch2,
                self.fbank_heatmaps_batch2,
                self.fbank_masks_batch2,
                self.tgt_embed_heatmaps_batch2,
                self.tgt_embed_masks_batch2)
            self.assertTrue(os.path.exists(file_path))
            with h5py.File(file_path, "r") as file:
                # check group 0
                group = file["0"]
                source_text = group["src_text"][()]
                tgt_text = group["tgt_text"][()]
                fbank_heatmap = group["fbank_heatmap"][()]
                tgt_embed_heatmap = group["tgt_embed_heatmap"][()]
                n_masks = group["n_masks"][()]
                self.assertEqual(source_text.decode('UTF-8'), "source text 1")
                self.assertEqual(
                    [x.decode('UTF-8') for x in tgt_text.tolist()], ["</s>", "_lui", "<unk>"])
                self.assertTrue(torch.equal(
                    torch.from_numpy(fbank_heatmap), torch.ones(3, 6, 10)))
                self.assertTrue(torch.equal(
                    torch.from_numpy(tgt_embed_heatmap), torch.ones(3, 3, 24)))
                self.assertEqual(n_masks, 2)
                # check group 1
                group = file["1"]
                source_text = group["src_text"][()]
                tgt_text = group["tgt_text"][()]
                fbank_heatmap = group["fbank_heatmap"][()]
                tgt_embed_heatmap = group["tgt_embed_heatmap"][()]
                n_masks = group["n_masks"][()]
                self.assertEqual(source_text.decode('UTF-8'), "source text 2")
                self.assertEqual(
                    [x.decode('UTF-8') for x in tgt_text.tolist()],
                    ["</s>", "<unk>", "_lui", "ava"])
                self.assertTrue(torch.equal(
                    torch.from_numpy(fbank_heatmap), torch.ones(4, 7, 10)))
                self.assertTrue(torch.equal(
                    torch.from_numpy(tgt_embed_heatmap), torch.ones(4, 4, 24)))
                self.assertEqual(n_masks, 3)
                # check group 2
                group = file["2"]
                source_text = group["src_text"][()]
                tgt_text = group["tgt_text"][()]
                fbank_heatmap = group["fbank_heatmap"][()]
                tgt_embed_heatmap = group["tgt_embed_heatmap"][()]
                n_masks = group["n_masks"][()]
                self.assertEqual(source_text.decode('UTF-8'), "source text 3")
                self.assertEqual(
                    [x.decode('UTF-8') for x in tgt_text.tolist()],
                    ["</s>", "ava", "<unk>", "_lui", "_and"])
                self.assertTrue(torch.equal(
                    torch.from_numpy(fbank_heatmap), torch.ones(5, 9, 10)))
                self.assertTrue(torch.equal(
                    torch.from_numpy(tgt_embed_heatmap), torch.ones(5, 5, 24)))
                self.assertEqual(n_masks, 2)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
