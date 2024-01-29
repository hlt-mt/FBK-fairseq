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

import unittest
import torch
from torch import Size

from examples.speech_to_text.occlusion_explanation.scorer import PredictedTokenDifferenceScorer, KLScorer


class TestScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = PredictedTokenDifferenceScorer()
        self.sample = {
            "id": [],
            "orig_id": torch.LongTensor([1, 1, 3]),
            "net_input": {
                "target": torch.tensor([[2, 1, 3, 5], [1, 1, 1, 2], [0, 1, 2, 5]]),
                "target_lengths": torch.LongTensor([7, 7, 9])},
            "masks": torch.zeros(3, 8, 7)}

    def test_get_padded_probs(self):
        orig_probs = {
            "0": torch.ones(6, 10), "1": torch.ones(7, 10), "3": torch.ones(9, 10)}
        perturb_probs = torch.ones(3, 9, 10)
        batched_orig_probs, batched_perturb_probs = self.scorer.get_padded_probs(
            orig_probs, perturb_probs, self.sample["orig_id"], self.sample["net_input"]["target_lengths"])
        expected_probs = torch.tensor(
            [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])
        self.assertTrue(torch.equal(batched_orig_probs, expected_probs))
        self.assertTrue(torch.equal(batched_perturb_probs, expected_probs))

    # Test get_padded_probs() with a target_length equal to 0
    def test_get_padded_probs_empty(self):
        orig_probs = {
            "0": torch.ones(2, 10), "1": torch.ones(3, 10), "3": torch.ones(4, 10), "4": torch.tensor([])}
        perturb_probs = torch.ones(2, 4, 10)
        target_lengths = torch.tensor([4, 0])
        orig_indices = torch.tensor([3, 4])
        batched_orig_probs, batched_perturb_probs = self.scorer.get_padded_probs(
            orig_probs, perturb_probs, orig_indices, target_lengths)
        expected_batched_probs = torch.tensor(
            [
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
        self.assertTrue(torch.equal(batched_orig_probs, expected_batched_probs))
        self.assertTrue(torch.equal(batched_perturb_probs, expected_batched_probs))

    def test_get_prob_diff_single(self):
        orig_probs = torch.tensor(
            [
                [
                    [2, 2, 2, 1, 2, 1],
                    [6, 4, 2, 1, 2, 1],
                    [8, 9, 2, 1, 1, 1],
                    [6, 7, 4, 2, 1, 1]],
                [
                    [2, 2, 2, 1, 2, 1],
                    [6, 4, 2, 1, 2, 1],
                    [8, 9, 2, 1, 1, 1],
                    [6, 7, 4, 2, 1, 1]],
                [
                    [2, 2, 2, 1, 2, 1],
                    [6, 4, 2, 1, 2, 1],
                    [8, 9, 2, 1, 1, 1],
                    [6, 7, 4, 2, 1, 1]]])
        perturb_probs = torch.tensor(
            [
                [
                    [1, 0, 1, 0, 0, 1],
                    [2, 0, 1, 3, 1, 3],
                    [1, 2, 1, 1, 3, 1],
                    [2, 0, 1, 3, 3, 2]],
                [
                    [1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 2, 1, 0],
                    [1, 1, 1, 1, 0, 1],
                    [3, 2, 3, 1, 1, 1]],
                [
                    [3, 1, 4, 1, 2, 2],
                    [0, 0, 0, 2, 1, 0],
                    [1, 3, 3, 2, 1, 2],
                    [2, 0, 0, 1, 1, 2]]])
        scores = self.scorer.get_prob_diff(orig_probs, perturb_probs, self.sample["net_input"]["target"])
        expected_scores = torch.tensor(
            [
                [
                    [1],
                    [4],
                    [0],
                    [-1]],
                [
                    [2],
                    [3],
                    [8],
                    [1]],
                [
                    [-1],
                    [4],
                    [-1],
                    [-1]]])
        self.assertTrue(torch.equal(scores, expected_scores))

    def test_get_prob_diff_KL(self):
        scorer = KLScorer()
        orig_probs = torch.rand((3, 8, 10))
        perturb_probs = torch.rand((3, 8, 10))
        scores = scorer.get_prob_diff(orig_probs, perturb_probs, self.sample["net_input"]["target"])
        self.assertEqual(scores.size(), Size([3, 8, 1]))

    def test_make_heatmaps_causal(self):
        # Batch size, sequence length, sequence length, embedding dimension
        heatmaps = torch.ones((1, 3, 3, 5))
        causality_heatmaps = self.scorer._make_heatmaps_causal(heatmaps)
        expected_heatmaps = torch.tensor(
            [[[[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]]])
        self.assertTrue(torch.equal(causality_heatmaps, expected_heatmaps))

        heatmaps = torch.ones((2, 4, 4, 5))
        causality_heatmaps = self.scorer._make_heatmaps_causal(heatmaps)
        expected_heatmaps = torch.tensor(
            [[[[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]],
             [[[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]]])
        self.assertTrue(torch.equal(causality_heatmaps, expected_heatmaps))

        heatmaps_empty = torch.empty((0, 0, 0, 0))
        causality_heatmaps_empty = self.scorer._make_heatmaps_causal(heatmaps_empty)
        self.assertTrue(torch.equal(causality_heatmaps_empty, heatmaps_empty))

    def test_get_heatmaps(self):
        _tgt_embed_masks = torch.zeros(3, 4, 6)
        scores = torch.tensor([[[1], [4], [0], [-1]], [[2], [3], [8], [1]], [[-1], [4], [-1], [-1]]])
        single_fbank_heatmaps, fbank_masks, single_tgt_embed_heatmaps, tgt_embed_masks = \
            self.scorer.get_heatmaps(scores, self.sample["masks"], _tgt_embed_masks)
        expected_single_fbank_heatmaps = torch.tensor(
            [
                [
                    [
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1]],
                    [
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4]],
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]],
                    [
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1]]],
                [
                    [
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2]],
                    [
                        [3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3, 3]],
                    [
                        [8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8]],
                    [
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1]]],
                [
                    [
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1]],
                    [
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4, 4]],
                    [
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1]],
                    [
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1]]]])
        expected_single_tgt_embed_heatmaps = torch.tensor(
            [
                [
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1]],
                    [
                        [4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4]],
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]],
                    [
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1]]],
                [
                    [
                        [2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2]],
                    [
                        [3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3, 3]],
                    [
                        [8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8]],
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1]]],
                [
                    [
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1]],
                    [
                        [4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4]],
                    [
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1]],
                    [
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1]]]])
        self.assertTrue(torch.equal(tgt_embed_masks, torch.ones(3, 4, 6).unsqueeze(1)))
        self.assertTrue(torch.equal(fbank_masks, torch.ones(3, 8, 7).unsqueeze(1)))
        self.assertTrue(torch.equal(single_fbank_heatmaps, expected_single_fbank_heatmaps))
        self.assertTrue(torch.equal(single_tgt_embed_heatmaps, expected_single_tgt_embed_heatmaps))


if __name__ == '__main__':
    unittest.main()
