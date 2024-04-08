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

from examples.speech_to_text.occlusion_explanation.aggregators.time import TimeLevelAggregator


class TestTimeLevelAggregators(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregator = TimeLevelAggregator()
        self.explanations = {
            0: {
                "fbank_heatmap": torch.tensor(  # (3, 4, 3)
                    [[[1., 0., 2.],
                      [4., 5., 6.],
                      [0., 2., 1.],
                      [2., 2., 2.]],
                     [[2., 0., 7.],
                      [1., 8., 9.],
                      [1., 1., 1.],
                      [1., 3., 2.]],
                     [[2., 1., 7.],
                      [2., 0., 9.],
                      [2., 2., 1.],
                      [2., 1., 2.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (3, 3, 1)
                    [[[1.], [0.], [0.]],
                     [[5.], [8.], [0.]],
                     [[2.], [5.], [6.]]]),
                "tgt_text": ['You', 'are', 'you']},
            1: {
                "fbank_heatmap": torch.tensor(  # (3, 3, 1)
                    [[[13.], [14.], [15.]],
                     [[19.], [20.], [21.]],
                     [[22.], [23.], [24.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (3, 3, 2)
                    [[[7., 8.], [5., 4.], [0., 0.]],
                     [[10., 2.], [11., 2.], [1., 2.]],
                     [[10., 2.], [11., 2.], [1., 2.]]]),
                "tgt_text": ['Clever', 'child', '.']}}

    def test_aggregator_with_indices(self):
        indices = {0: [(0, 1), (1, 3)], 1: [(1, 1), (2, 2)]}
        aggregated_explanations = self.aggregator(self.explanations, indices)

        expected_output = {
            0: {
                "fbank_heatmap": torch.tensor(  # (3, 2, 3)
                    [[[2.5, 2.5, 4.], [2., 3., 3.]],
                     [[1.5, 4., 8.], [1., 4., 4.]],
                     [[2., 0.5, 8.], [2., 1., 4.]]]),
                "tgt_embed_heatmap": torch.tensor(
                    [[[1.], [0.], [0.]],
                     [[5.], [8.], [0.]],
                     [[2.], [5.], [6.]]]),
                "tgt_text": ['You', 'are', 'you']},
            1: {
                "fbank_heatmap": torch.tensor(  # (3, 2, 1)
                    [[[14.], [15.]],
                     [[20.], [21.]],
                     [[23.], [24.]]]),
                "tgt_embed_heatmap": torch.tensor(
                    [[[7., 8.], [5., 4.], [0., 0.]],
                     [[10., 2.], [11., 2.], [1., 2.]],
                     [[10., 2.], [11., 2.], [1., 2.]]]),
                "tgt_text": ['Clever', 'child', '.']}}

        self.assertEqual([0, 1], list(aggregated_explanations.keys()))
        for sample_id in aggregated_explanations:
            self.assertTrue(torch.equal(
                expected_output[sample_id]["fbank_heatmap"],
                aggregated_explanations[sample_id]["fbank_heatmap"]))
            self.assertTrue(torch.equal(
                expected_output[sample_id]["tgt_embed_heatmap"],
                aggregated_explanations[sample_id]["tgt_embed_heatmap"]))
            self.assertEqual(
                expected_output[sample_id]["tgt_text"], aggregated_explanations[sample_id]["tgt_text"])

    def test_aggregator_without_indices(self):
        aggregated_explanations = self.aggregator(self.explanations)

        expected_output = {
            0: {
                "fbank_heatmap": torch.tensor(  # (3, 1, 3)
                    [[[1.75, 2.25, 2.75]],
                     [[1.25, 3., 4.75]],
                     [[2., 1., 4.75]]]),
                "tgt_embed_heatmap": torch.tensor(
                    [[[1.], [0.], [0.]],
                     [[5.], [8.], [0.]],
                     [[2.], [5.], [6.]]]),
                "tgt_text": ['You', 'are', 'you']},
            1: {
                "fbank_heatmap": torch.tensor([[[14.]], [[20.]], [[23.]]]),  # (3, 1, 1)
                "tgt_embed_heatmap": torch.tensor(
                    [[[7., 8.], [5., 4.], [0., 0.]],
                     [[10., 2.], [11., 2.], [1., 2.]],
                     [[10., 2.], [11., 2.], [1., 2.]]]),
                "tgt_text": ['Clever', 'child', '.']}}

        self.assertEqual([0, 1], list(aggregated_explanations.keys()))
        for sample_id in aggregated_explanations:
            self.assertTrue(torch.equal(
                expected_output[sample_id]["fbank_heatmap"],
                aggregated_explanations[sample_id]["fbank_heatmap"]))
            self.assertTrue(torch.equal(
                expected_output[sample_id]["tgt_embed_heatmap"],
                aggregated_explanations[sample_id]["tgt_embed_heatmap"]))
            self.assertEqual(
                expected_output[sample_id]["tgt_text"], aggregated_explanations[sample_id]["tgt_text"])


if __name__ == "__main__":
    unittest.main()
