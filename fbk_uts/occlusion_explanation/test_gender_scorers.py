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

from examples.speech_to_text.occlusion_explanation.scorers.gender_term_contrastive import GenderTermContrastiveScorer
from examples.speech_to_text.occlusion_explanation.scorers.kl_gender_terms import KLGenderScorer


class TestGenderScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.KL_scorer = KLGenderScorer()
        self.contrastive_scorer = GenderTermContrastiveScorer()
        self.sample = {
            "id": [],
            "orig_id": torch.LongTensor([1, 1]),
            "target": torch.tensor([[2, 1, 3], [1, 1, 1]]),
            "target_lengths": torch.LongTensor([3, 3]),
            "swapped_target": torch.tensor([[2, 1, 3, 0], [1, 1, 2, 1]]),
            "swapped_target_lengths": torch.LongTensor([3, 4]),
            "gender_terms_indices": ['0-1', '1-1']}
        self.orig_probs = torch.tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
              [0.6, 0.4, 0.2, 0.1, 0.2, 0.1],
              [0.8, 0.9, 0.2, 0.1, 0.1, 0.1]],
             [[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
              [0.6, 0.4, 0.2, 0.1, 0.2, 0.1],
              [0.8, 0.9, 0.2, 0.1, 0.1, 0.1]]]) # (Batch size, sequence length, vocab size)
        self.perturb_probs = torch.tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.2, 0.1, 0.1, 0.3, 0.1, 0.3],
              [0.1, 0.2, 0.1, 0.1, 0.3, 0.1]],
             [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]) # (Batch size, sequence length, vocab size)
        self.swapped_orig_probs = torch.tensor(
            [[[0.1, 0.01, 0.1, 0.1, 0.1, 0.1],
              [0.02, 0.1, 0.01, 0.2, 0.1, 0.2],
              [0.1, 0.02, 0.1, 0.1, 0.2, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.2, 0.1]],
             [[0.01, 0.02, 0.01, 0.1, 0.1, 0.1],
              [0.1, 0.02, 0.1, 0.02, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.2, 0.1]]]) # (Batch size, sequence length, vocab size)
        self.swapped_perturb_probs = torch.tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.2, 0.01, 0.1, 0.3, 0.1, 0.3],
              [0.1, 0.02, 0.1, 0.1, 0.3, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.3, 0.1]],
             [[0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
              [0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.02, 0.1, 0.1, 0.3, 0.1]]]) # (Batch size, sequence length, vocab size)
        
    def test_get_prob_diff_KL(self):
        scores = self.KL_scorer.get_prob_diff(self.orig_probs, self.perturb_probs, self.sample)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[1.2712]], [[1.8375]]])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.0001))

    def test_get_prob_diff_contrastive(self):
        scores = self.contrastive_scorer.get_prob_diff(
            self.orig_probs, self.perturb_probs, self.swapped_orig_probs,
            self.swapped_perturb_probs, self.sample)
        self.assertEqual(scores.size(), Size([2, 1, 1]))
        expected_scores = torch.tensor([[[0.11446]], [[0.35528]]])    # (Batch size, 1, 1)
        self.assertTrue(torch.allclose(scores, expected_scores, atol=0.00001))

    def test_make_heatmaps_causal(self):
        # Batch size, gender term length, sequence length, embedding dimension
        heatmaps = torch.ones((2, 1, 3, 5))
        causality_heatmaps = self.KL_scorer._make_heatmaps_causal(heatmaps, self.sample)
        expected_heatmaps = torch.tensor(
            [[[[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]],
             [[[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.]]]])
        self.assertTrue(torch.equal(causality_heatmaps, expected_heatmaps))

    # test make_heatmaps_causal() when the masking strategy is 'discrete'
    def test_make_heatmaps_causal_discrete(self):
        heatmaps = torch.ones((2, 1, 3, 1))  # (Batch size, sequence length, sequence length, 1)
        causality_heatmaps = self.KL_scorer._make_heatmaps_causal(heatmaps, self.sample)
        expected_heatmaps = torch.tensor(
            [[[[1.], [0.], [0.]]],
             [[[1.], [1.], [0.]]]])
        self.assertTrue(torch.equal(causality_heatmaps, expected_heatmaps))


if __name__ == '__main__':
    unittest.main()
