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

from examples.speech_to_text.occlusion_explanation.aggregators.sentence_level import (
    SentenceLevelAggregatorMinMaxNormalization,
    SentenceLevelAggregatorNoNormalization,
    SentenceLevelAggregatorMeanStdNormalization)


class TestSentenceLevelAggregators(unittest.TestCase):
    def setUp(self) -> None:
        self.explanations = {
            0: {
                "fbank_heatmap": torch.tensor(
                    [[[1., 5., 1.], [1., 5., 1.]], [[0., 2., 0.], [0., 2., 0.]]]),  # (2, 2, 3)
                "tgt_embed_heatmap": torch.tensor(
                    [[[5., 1., 5.], [5., 1., 5.]], [[2., 0., 2.], [2., 0., 2.]]]),  # (2, 2, 3)
                "src": "sentence 1."},
            1: {
                "fbank_heatmap": torch.tensor([[[1., 5., 1.], [1., 5., 1.]]]),  # (1, 2, 3)
                "tgt_embed_heatmap": torch.tensor([[[0., 1., 0., 5.]]]),  # (1, 1, 4)
                "src": "sentence 2."},
            2: {
                "fbank_heatmap": torch.tensor([[[1., 5., 1., 5.]], [[1., 5., 1., 5.]]]),  # (2, 1, 4)
                "tgt_embed_heatmap": torch.tensor([[[1.], [2.]], [[1.], [2.]]]),  # (2, 2, 1)
                "src": "sentence 3."},
            3: {
                "fbank_heatmap": torch.tensor([[[10., 20.]]]),  # (1, 1, 2)
                "tgt_embed_heatmap": torch.tensor([[[100.]]]),  # (1, 1, 1)
                "src": "sentence 4."}}

    # test _normalize() of SentenceLevelAggregatorNoNormalization()
    def test_no_norm_normalize(self):
        aggregator = SentenceLevelAggregatorNoNormalization()
        fbank_norm, tgt_norm = aggregator._normalize(
            self.explanations[3]["fbank_heatmap"], self.explanations[3]["tgt_embed_heatmap"])
        self.assertTrue(torch.equal(fbank_norm, torch.tensor([[[10, 20]]])))
        self.assertTrue(torch.equal(tgt_norm, torch.tensor([[[100]]])))

    # test __call__() of SentenceLevelAggregatorNoNormalization()
    def test_no_norm_call(self):
        aggregator = SentenceLevelAggregatorNoNormalization()
        aggregated_explanations = aggregator(self.explanations)
        self.assertTrue(torch.equal(
            aggregated_explanations[0][0], torch.tensor([[1, 7, 1], [1, 7, 1]])))
        self.assertTrue(torch.equal(
            aggregated_explanations[0][1], torch.tensor([[7, 1, 7], [7, 1, 7]])))

    # test _normalize() of SentenceLevelAggregatorMinMaxNormalization()
    def test_min_max_normalization(self):
        aggregator = SentenceLevelAggregatorMinMaxNormalization()
        fbank_norm, tgt_norm = aggregator._normalize(
            self.explanations[1]["fbank_heatmap"], self.explanations[1]["tgt_embed_heatmap"])
        expected_fbank_norm = torch.tensor(
            [[[0.2, 1.0, 0.2], [0.2, 1.0, 0.2]]])
        expected_tgt_norm = torch.tensor([[[0, 0.2, 0, 1]]])
        self.assertTrue(torch.equal(fbank_norm, expected_fbank_norm))
        self.assertTrue(torch.equal(tgt_norm, expected_tgt_norm))

    # test __call__() of SentenceLevelAggregatorMinMaxNormalization()
    def test_min_max_call(self):
        aggregator = SentenceLevelAggregatorMinMaxNormalization()
        aggregated_explanations = aggregator(self.explanations)
        self.assertTrue(torch.equal(
            aggregated_explanations[1][0], torch.tensor([[0.2, 1.0, 0.2], [0.2, 1.0, 0.2]])))
        self.assertTrue(torch.equal(
            aggregated_explanations[1][1], torch.tensor([[0, 0.2, 0, 1]])))

    # test _normalize() of SentenceLevelAggregatorMeanStdNormalization()
    def test_mean_std_normalization(self):
        aggregator = SentenceLevelAggregatorMeanStdNormalization()
        fbank_norm, tgt_norm = aggregator._normalize(
            self.explanations[2]["fbank_heatmap"], self.explanations[2]["tgt_embed_heatmap"])
        print(fbank_norm)
        print(tgt_norm)
        expected_fbank_norm = torch.tensor(
            [[[-0.7596, 1.2659, -0.7596, 1.2659]], [[-0.7596, 1.2659, -0.7596, 1.2659]]])
        expected_tgt_norm = torch.tensor([[[-0.7596], [-0.2532]], [[-0.7596], [-0.2532]]])
        print(expected_fbank_norm)
        print(expected_tgt_norm)
        self.assertTrue(torch.allclose(fbank_norm, expected_fbank_norm, atol=1e-04))
        self.assertTrue(torch.allclose(tgt_norm, expected_tgt_norm, atol=1e-04))

    # test __call__() of SentenceLevelAggregatorMeanStdNormalization()
    def test_mean_std_call(self):
        aggregator = SentenceLevelAggregatorMeanStdNormalization()
        aggregated_explanations = aggregator(self.explanations)
        self.assertTrue(torch.allclose(
            aggregated_explanations[2][0], torch.tensor([[-1.5192, 2.5318, -1.5192, 2.5318]]), atol=1e-04))
        self.assertTrue(torch.allclose(
            aggregated_explanations[2][1], torch.tensor([[-1.5192], [-0.5064]]), atol=1e-04))


if __name__ == '__main__':
    unittest.main()
