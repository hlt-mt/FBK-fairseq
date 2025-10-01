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

from examples.speech_to_text.occlusion_explanation.scorers.gender_term_contrastive_cross_ratio import GenderTermContrastiveCrossRatioScorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_term_contrastive_parity_ratio import GenderTermContrastiveParityRatioScorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_term_contrastive_ratio_difference import GenderTermContrastiveRatioDiffScorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_term_contrastive_raw_diff import GenderTermContrastiveRawDiffScorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_term_predicted_diff import GenderTermPredictedDiffScorer
from examples.speech_to_text.occlusion_explanation.scorers.gender_term_predicted_ratio import GenderTermPredictedRatioScorer
from examples.speech_to_text.occlusion_explanation.scorers.probability_aggregators import get_prob_aggregator


class TestGenderScorer(unittest.TestCase):
    def setUp(self) -> None:
        length_norm_aggregator = get_prob_aggregator("length_norm")()
        self.contrastive_raw_diff_scorer = GenderTermContrastiveRawDiffScorer(length_norm_aggregator)
        self.contrastive_ratio_diff_scorer = GenderTermContrastiveRatioDiffScorer(length_norm_aggregator)
        self.contrastive_cross_ratio_scorer = GenderTermContrastiveCrossRatioScorer(length_norm_aggregator)
        self.contrastive_parity_ratio_scorer = GenderTermContrastiveParityRatioScorer(length_norm_aggregator)
        self.diff_scorer = GenderTermPredictedDiffScorer(length_norm_aggregator)
        self.ratio_scorer = GenderTermPredictedRatioScorer(length_norm_aggregator)       
        self.sample = {
            "id": [],
            "orig_id": torch.LongTensor([1, 1]),
            "target": torch.tensor([[2, 1, 3], [1, 1, 1]]),
            "target_lengths": torch.LongTensor([3, 3]),
            "swapped_target": torch.tensor([[2, 1, 3, 0], [1, 1, 2, 1]]),
            "swapped_target_lengths": torch.LongTensor([3, 4]),
            "gender_term_starts": torch.LongTensor([0, 1]),
            "gender_term_ends": torch.LongTensor([1, 1]),
            "swapped_term_ends": torch.LongTensor([1, 2])}
        self.orig_probs = torch.tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
              [0.6, 0.4, 0.2, 0.1, 0.2, 0.1],
              [0.8, 0.9, 0.2, 0.1, 0.1, 0.1]],
             [[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
              [0.6, 0.4, 0.2, 0.1, 0.2, 0.1],
              [0.8, 0.9, 0.2, 0.1, 0.1, 0.1]]]) # (Batch size, sequence length, vocab size)
        self.gt_orig_probs = torch.tensor([[[0.2828]], [[0.4000]]])
        self.perturb_probs = torch.tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.2, 0.1, 0.1, 0.3, 0.1, 0.3],
              [0.1, 0.2, 0.1, 0.1, 0.3, 0.1]],
             [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]) # (Batch size, sequence length, vocab size)
        self.gt_perturb_probs = torch.tensor([[[0.1000]], [[0.1000]]])
        self.swapped_orig_probs = torch.tensor(
            [[[0.1, 0.01, 0.1, 0.1, 0.1, 0.1],
              [0.02, 0.1, 0.01, 0.2, 0.1, 0.2],
              [0.1, 0.02, 0.1, 0.1, 0.2, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.2, 0.1]],
             [[0.01, 0.02, 0.01, 0.1, 0.1, 0.1],
              [0.1, 0.02, 0.1, 0.02, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.2, 0.1]]]) # (Batch size, sequence length, vocab size)
        self.swapped_orig_gt_probs = torch.tensor([[[0.1]], [[0.0447]]])
        self.swapped_perturb_probs = torch.tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.2, 0.01, 0.1, 0.3, 0.1, 0.3],
              [0.1, 0.02, 0.1, 0.1, 0.3, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.3, 0.1]],
             [[0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
              [0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.3, 0.1]]]) # (Batch size, sequence length, vocab size)
        self.swapped_perturb_gt_probs = torch.tensor([[[0.0316]], [[0.1]]])

    def test_get_prob_diff_predicted_diff(self):
        scores = self.diff_scorer.get_prob_diff(self.gt_orig_probs, self.gt_perturb_probs)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[0.18284]], [[0.3]]])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.0001))

    def test_get_prob_diff_predicted_ratio(self):
        scores = self.ratio_scorer.get_prob_diff(
            self.gt_orig_probs, self.gt_perturb_probs)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[2.828]], [[4.]]])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.0001))

    def test_get_prob_diff_contrastive(self):
        scores = self.contrastive_raw_diff_scorer.get_prob_diff(
            self.gt_orig_probs,
            self.gt_perturb_probs,
            self.swapped_orig_gt_probs,
            self.swapped_perturb_gt_probs)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[0.1144]], [[0.3553]]])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.0001))

    def test_get_prob_diff_contrastive_ratio_diff(self):
        scores = self.contrastive_ratio_diff_scorer.get_prob_diff(
            self.gt_orig_probs,
            self.gt_perturb_probs,
            self.swapped_orig_gt_probs,
            self.swapped_perturb_gt_probs)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[-0.33656]], [[3.553]]])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.0001))

    def test_get_prob_diff_contrastive_cross_ratio(self):
        scores = self.contrastive_cross_ratio_scorer.get_prob_diff(
            self.gt_orig_probs,
            self.gt_perturb_probs,
            self.swapped_orig_gt_probs,
            self.swapped_perturb_gt_probs)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[-0.33656]], [[7.9485]]])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.0001))

    def test_get_prob_diff_contrastive_parity_ratio(self):
        scores = self.contrastive_parity_ratio_scorer.get_prob_diff(
            self.gt_orig_probs,
            self.gt_perturb_probs,
            self.swapped_orig_gt_probs,
            self.swapped_perturb_gt_probs)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[-0.021111]], [[0.39948]]])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.0001))

    def test_make_heatmaps_causal(self):
        # Batch size, gender term length, sequence length, embedding dimension
        heatmaps = torch.ones((2, 1, 3, 5))
        causality_heatmaps = self.contrastive_raw_diff_scorer._make_heatmaps_causal(heatmaps, self.sample)
        expected_heatmaps = torch.tensor(
            [[[[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]],
             [[[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.]]]])
        self.assertTrue(torch.equal(causality_heatmaps, expected_heatmaps))

    # test make_heatmaps_causal() when the masking strategy is 'discrete'
    def test_make_heatmaps_causal_discrete(self):
        heatmaps = torch.ones((2, 1, 3, 1))  # (Batch size, sequence length, sequence length, 1)
        causality_heatmaps = self.contrastive_raw_diff_scorer._make_heatmaps_causal(heatmaps, self.sample)
        expected_heatmaps = torch.tensor(
            [[[[1.], [0.], [0.]]],
             [[[1.], [1.], [0.]]]])
        self.assertTrue(torch.equal(causality_heatmaps, expected_heatmaps))


if __name__ == '__main__':
    unittest.main()
