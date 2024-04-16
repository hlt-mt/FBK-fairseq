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

from examples.speech_to_text.occlusion_explanation.aggregators.sentence import SentenceLevelAggregator


class TestSentenceLevelAggregator(unittest.TestCase):
    def setUp(self) -> None:
        self.explanations = {
            0: {
                "fbank_heatmap": torch.tensor(
                    [[[1., 5., 1.], [1., 5., 1.]], [[0., 2., 0.], [0., 2., 0.]]]),  # (2, 2, 3)
                "tgt_embed_heatmap": torch.tensor(
                    [[[5., 1., 5.], [5., 1., 5.]], [[2., 0., 2.], [2., 0., 2.]]]),  # (2, 2, 3)
                "tgt_text": "sentence 1."},
            1: {
                "fbank_heatmap": torch.tensor([[[1., 5., 1.], [1., 5., 1.]]]),  # (1, 2, 3)
                "tgt_embed_heatmap": torch.tensor([[[0., 1., 0., 5.]]]),  # (1, 1, 4)
                "tgt_text": "sentence 2."},
            2: {
                "fbank_heatmap": torch.tensor([[[1., 5., 1., 5.]], [[1., 5., 1., 5.]]]),  # (2, 1, 4)
                "tgt_embed_heatmap": torch.tensor([[[1.], [2.]], [[1.], [2.]]]),  # (2, 2, 1)
                "tgt_text": "sentence 3."},
            3: {
                "fbank_heatmap": torch.tensor([[[10., 20.]]]),  # (1, 1, 2)
                "tgt_embed_heatmap": torch.tensor([[[100.]]]),  # (1, 1, 1)
                "tgt_text": "sentence 4."}}

    def test_call(self):
        aggregator = SentenceLevelAggregator()
        aggregated_explanations = aggregator(self.explanations)
        self.assertTrue(torch.equal(
            aggregated_explanations[0]["fbank_heatmap"], torch.tensor([[[0.5, 3.5, 0.5], [0.5, 3.5, 0.5]]])))
        self.assertTrue(torch.equal(
            aggregated_explanations[0]["tgt_embed_heatmap"], torch.tensor([[[3.5, 0.5, 3.5], [3.5, 0.5, 3.5]]])))
        self.assertEqual(aggregated_explanations[0]["tgt_text"], self.explanations[0]["tgt_text"])


if __name__ == '__main__':
    unittest.main()
