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

from examples.speech_to_text.occlusion_explanation.aggregators.frequency_embed import FrequencyEmbeddingAggregator


class TestTimeLevelAggregators(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregator = FrequencyEmbeddingAggregator()
        self.explanations = {
            0: {
                "fbank_heatmap": torch.tensor(  # (6, 2, 3)
                    [[[1., 5., 0.],
                      [1., 5., 3.]],
                     [[0., 2., 1.],
                      [1., 2., 0.]],
                     [[2., 5., 2.],
                      [1., 5., 3.]],
                     [[1., 2., 0.],
                      [0., 3., 0.]],
                     [[1., 6., 2.],
                      [1., 4., 1.]],
                     [[1., 2., 0.],
                      [0., 2., 1.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (6, 6, 1)
                    [[[2.], [0.], [0.], [0.], [0.], [0.]],
                     [[3.], [2.5], [0.], [0.], [0.], [0.]],
                     [[1.], [1.6], [1.], [0.], [0.], [0.]],
                     [[1.], [1.6], [1.], [1.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [0.]],
                     [[0.5], [1.5], [0.5], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', '▁Swim', 'm', 'ing', '▁is', '▁hard']},
            1: {
                "fbank_heatmap": torch.tensor(  # (7, 4, 1)
                    [[[1.], [5.], [2.], [1.]],
                     [[2.], [1.], [1.], [3.]],
                     [[1.], [1.], [1.], [0.]],
                     [[2.], [2.5], [1.5], [0.]],
                     [[0.], [3.5], [1.], [1.]],
                     [[1.], [2.], [1.], [2.]],
                     [[2.], [1.], [2.], [1.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (7, 7, 2)
                    [[[1., 2.], [0., 0.], [0., 0.], [0., 2.], [0., 2.], [0., 2.], [0., 5.]],
                     [[1., 3.], [1., 0.], [0., 0.], [0., 0.], [0., 5.], [0., 2.], [0., 0.]],
                     [[1., 3.], [1., 5.], [1., 1.], [1., 0.], [4., 0.], [4., 0.], [4., 0.]],
                     [[0., 1.], [0., 1.], [1., 2.], [1., 0.], [0., 5.], [0., 4.], [0., 0.]],
                     [[1., 0.], [1., 2.], [1., 3.], [1., 1.], [1., 3.], [0., 5.], [0., 5.]],
                     [[1., 0.], [2., 1.], [1., 5.], [1., 1.], [1., 0.], [1., 0.], [0., 5.]],
                     [[1., 5.], [1., 5.], [2., 1.], [1., 2.], [1., 3.], [1., 5.], [1., 5.]]]),
                "tgt_text": ['</s>', '▁You', "'", 're', '▁co', 'rrect', '.']}}

    def test_call(self):
        expected = {
            0: {
                "fbank_heatmap": torch.tensor(  # (6, 2, 1)
                    [[[2.], [3.]],
                     [[1.], [1.]],
                     [[3.], [3.]],
                     [[1.], [1.]],
                     [[3.], [2.]],
                     [[1.], [1.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (6, 6, 1)
                    [[[2.], [0.], [0.], [0.], [0.], [0.]],
                     [[3.], [2.5], [0.], [0.], [0.], [0.]],
                     [[1.], [1.6], [1.], [0.], [0.], [0.]],
                     [[1.], [1.6], [1.], [1.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [0.]],
                     [[0.5], [1.5], [0.5], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', '▁Swim', 'm', 'ing', '▁is', '▁hard']},
            1: {
                "fbank_heatmap": torch.tensor(  # (7, 4, 1)
                    [[[1.], [5.], [2.], [1.]],
                     [[2.], [1.], [1.], [3.]],
                     [[1.], [1.], [1.], [0.]],
                     [[2.], [2.5], [1.5], [0.]],
                     [[0.], [3.5], [1.], [1.]],
                     [[1.], [2.], [1.], [2.]],
                     [[2.], [1.], [2.], [1.]]]),
                "tgt_embed_heatmap": torch.tensor(
                    [[[1.5], [0.], [0.], [1.], [1.], [1.], [2.5]],
                     [[2.], [0.5], [0.], [0.], [2.5], [1.], [0.]],
                     [[2.], [3.], [1.], [0.5], [2.], [2.], [2.]],
                     [[0.5], [0.5], [1.5], [0.5], [2.5], [2.], [0.]],
                     [[0.5], [1.5], [2.], [1.], [2.], [2.5], [2.5]],
                     [[0.5], [1.5], [3.], [1.], [0.5], [0.5], [2.5]],
                     [[3.], [3.], [1.5], [1.5], [2.], [3.], [3.]]]),
                "tgt_text": ['</s>', '▁You', "'", 're', '▁co', 'rrect', '.']}}

        aggregated_explanations = self.aggregator(self.explanations)
        for sample_id in expected.keys():
            self.assertTrue(torch.equal(
                aggregated_explanations[sample_id]["fbank_heatmap"],
                expected[sample_id]["fbank_heatmap"]))
            self.assertTrue(torch.equal(
                aggregated_explanations[sample_id]["tgt_embed_heatmap"],
                expected[sample_id]["tgt_embed_heatmap"]))
            self.assertEqual(
                aggregated_explanations[sample_id]["tgt_text"],
                expected[sample_id]["tgt_text"])


if __name__ == '__main__':
    unittest.main()
