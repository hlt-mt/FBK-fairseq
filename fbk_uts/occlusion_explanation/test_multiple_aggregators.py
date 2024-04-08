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
from examples.speech_to_text.occlusion_explanation.aggregators.phrase import PhraseAndWordLevelAggregator
from examples.speech_to_text.occlusion_explanation.aggregators.sentence import SentenceLevelAggregator


class TestMultipleAggregators(unittest.TestCase):
    """
    Test multiple aggregators used in a composite manner.
    """
    def setUp(self) -> None:
        self.word_aggregator = PhraseAndWordLevelAggregator()
        self.freq_embed_aggregator = FrequencyEmbeddingAggregator()
        self.sentence_aggregator = SentenceLevelAggregator()
        self.explanations = {
            0: {
                "fbank_heatmap": torch.tensor(  # (6, 2, 3)
                    [[[1., 5., 1.],
                      [1., 5., 1.]],
                     [[0., 2., 0.],
                      [1., 2., 1.]],
                     [[2., 5., 2.],
                      [1., 5., 1.]],
                     [[0., 2., 0.],
                      [0., 2., 0.]],
                     [[1., 5., 1.],
                      [1., 5., 1.]],
                     [[0., 2., 0.],
                      [0., 2., 0.]]]),
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
                "tgt_embed_heatmap": torch.tensor(  # (7, 7, 1)
                    [[[1.], [0.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [1.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', '▁You', "'", 're', '▁co', 'rrect', '.']}}


    # test word-level + frequency/embdding-level aggregators
    def test_word_and_freq(self):
        aggregated_explanations = self.word_aggregator(self.explanations)
        aggregated_explanations = self.freq_embed_aggregator(aggregated_explanations)
        expected = {
            0: {
                "fbank_heatmap": torch.tensor(  # (4, 2, 1)
                    [[[2.], [2.]], [[0.6667], [0.6667]], [[2.3334], [2.3334]], [[0.6667], [0.6667]]]),
                "tgt_embed_heatmap": torch.tensor(  # (4, 4, 1)
                    [[[2.], [0.], [0.], [0.]],
                     [[1.], [1.2], [0.], [0.]],
                     [[1.], [1.], [1.], [0.]],
                     [[0.5], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', 'Swimming', 'is', 'hard'],
                "tgt_phrases": ['</s>', 'Swimming', 'is', 'hard']},
            1: {
                "fbank_heatmap": torch.tensor(  # (6, 6, 1)
                    [[[1.], [5.], [2.], [1.]],
                     [[2.], [1.], [1.], [3.]],
                     [[1.], [1.], [1.], [0.]],
                     [[1.], [3], [1.25], [0.5]],
                     [[1.], [2.], [1.], [2.]],
                     [[2.], [1.], [2.], [1.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (6, 6, 1)
                    [[[1.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', 'You', "'", 're', 'correct', '.'],
                "tgt_phrases": ['</s>', 'You', "'", 're', 'correct', '.']}}

        self.assertEqual(list(expected.keys()), list(aggregated_explanations.keys()))
        for i in aggregated_explanations:
            self.assertEqual(
                list(aggregated_explanations[i].keys()),
                ["fbank_heatmap", "tgt_embed_heatmap", "tgt_text", "tgt_phrases"])
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["fbank_heatmap"], expected[i]["fbank_heatmap"], atol=0.0001))
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["tgt_embed_heatmap"], expected[i]["tgt_embed_heatmap"], atol=0.0001))
            self.assertEqual(aggregated_explanations[i]["tgt_text"], expected[i]["tgt_text"])
            self.assertEqual(aggregated_explanations[i]["tgt_phrases"], expected[i]["tgt_phrases"])

    # test word-level + sentence-level aggregators
    def test_word_and_sentence(self):
        aggregated_explanations = self.word_aggregator(self.explanations)
        aggregated_explanations = self.sentence_aggregator(aggregated_explanations)

        expected = {
            0: {
                "fbank_heatmap": torch.tensor(  # (1, 2, 3)
                    [[[0.5, 3.25, 0.5],
                      [0.5, 3.25, 0.5]]]),
                "tgt_embed_heatmap": torch.tensor(  # (1, 4, 1)
                    [[[1.125], [0.8], [0.5], [0.25]]]),
                "tgt_text": ['</s>', 'Swimming', 'is', 'hard'],
                "tgt_phrases": ['</s>', 'Swimming', 'is', 'hard']},
            1: {
                "fbank_heatmap": torch.tensor(  # (1, 4, 1)
                    [[[1.3333], [2.1667], [1.375], [1.25]]]),
                "tgt_embed_heatmap": torch.tensor(  # (1, 6, 1)
                    [[[1.], [0.8333], [0.6667], [0.5], [0.3333], [0.1667]]]),
                "tgt_text": ['</s>', 'You', "'", 're', 'correct', '.'],
                "tgt_phrases": ['</s>', 'You', "'", 're', 'correct', '.']}}

        self.assertEqual(list(expected.keys()), list(aggregated_explanations.keys()))
        for i in aggregated_explanations:
            self.assertEqual(
                list(aggregated_explanations[i].keys()),
                ["fbank_heatmap", "tgt_embed_heatmap", "tgt_text", "tgt_phrases"])
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["fbank_heatmap"], expected[i]["fbank_heatmap"], atol=0.0001))
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["tgt_embed_heatmap"], expected[i]["tgt_embed_heatmap"], atol=0.0001))
            self.assertEqual(aggregated_explanations[i]["tgt_text"], expected[i]["tgt_text"])
            self.assertEqual(aggregated_explanations[i]["tgt_phrases"], expected[i]["tgt_phrases"])


if __name__ == '__main__':
    unittest.main()
