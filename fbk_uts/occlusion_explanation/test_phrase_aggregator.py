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
import copy
import torch

from examples.speech_to_text.occlusion_explanation.aggregators.phrase import \
    PhraseLevelAggregator, PhraseAndWordLevelAggregator


MOCK_EXPLANATIONS = {
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


class TestPhraseLevelAggregator(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregator = PhraseLevelAggregator()

    def test_get_words(self):
        sentences = [
            ['▁who', '▁wants', '▁to', '▁do', '▁that', '.'],
            ['</s>', '▁And', '▁he', '▁says', ',', '▁"', 'We', "'", 're', '▁witness', 'ing', '.', '▁And'],
            ['▁he', "'", 's', '▁an', '▁economist'],
            ['</s>', '▁What', '▁we', '▁have', '▁to', '▁do', '.', '▁What', '▁am', '▁I', '▁app', 'eal', 'ing', '?'],
            ['▁let', "'", 's', '▁take', '▁a', '▁look', '.'],
            []]
        expected_words = [
            ['who', 'wants', 'to', 'do', 'that', '.'],
            ['</s>', 'And', 'he', 'says', ',', '"', 'We', "'", 're', 'witnessing', '.', 'And'],
            ['he', "'", 's', 'an', 'economist'],
            ['</s>', 'What', 'we', 'have', 'to', 'do', '.', 'What', 'am', 'I', 'appealing', '?'],
            ['let', "'", 's', 'take', 'a', 'look', '.'],
            []]
        expected_indices = [
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 10), (11, 11), (12, 12)],
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 12), (13, 13)],
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)],
            []]
        for s, e, i in zip(sentences, expected_words, expected_indices):
            words, word_indices = self.aggregator.get_words(s)
            self.assertEqual(words, e)
            self.assertEqual(word_indices, i)

    def test_aggregate_fbank_explanations_discrete(self):
        fbank_explanation = torch.tensor(
            [[[3], [0], [1], [3]],
             [[2], [1], [0.5], [2]],
             [[1], [3.5], [1.5], [1]],
             [[3], [5], [4], [2]],
             [[6], [1], [2], [2]]])
        expected1 = torch.tensor(
            [[[2], [1.5], [1], [2]],
             [[3], [5], [4], [2]],
             [[6], [1], [2], [2]]])
        expected3 = torch.tensor(
            [[[3], [0], [1], [3]],
             [[2], [4.25], [2.75], [1.5]]])
        expected4 = torch.tensor(
            [[[2], [4.25], [2.75], [1.5]],
             [[6], [1], [2], [2]]])

        aggregated_fbank1 = self.aggregator.aggregate_fbank_explanations(
            aggregation_indices=[(0, 0), (1, 3), (4, 4)],
            fbank_heatmap=fbank_explanation,
            include_eos=True)
        aggregated_fbank2 = self.aggregator.aggregate_fbank_explanations(
            aggregation_indices=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            fbank_heatmap=fbank_explanation,
            include_eos=True)
        aggregated_fbank3 = self.aggregator.aggregate_fbank_explanations(
            aggregation_indices=[(1, 1), (3, 4)],
            fbank_heatmap=fbank_explanation,
            include_eos=False)
        aggregated_fbank4 = self.aggregator.aggregate_fbank_explanations(
            aggregation_indices=[(0, 0), (3, 4)],
            fbank_heatmap=fbank_explanation,
            include_eos=True)

        self.assertTrue(torch.equal(aggregated_fbank1, expected1))
        self.assertTrue(torch.equal(aggregated_fbank2, fbank_explanation))
        self.assertTrue(torch.equal(aggregated_fbank3, expected3))
        self.assertTrue(torch.equal(aggregated_fbank4, expected4))

    def test_aggregate_fbank_explanations_continuous(self):
        fbank_explanation = torch.tensor(
            [[[2, 0, 0, 2], [1, 1, 4, 0], [1, 0, 1, 0.5]],
             [[1, 0, 0, 2], [1, 0, 3, 0], [1, 0, 1, 1.5]],
             [[3, 3, 0, 2], [1, 5, 8, 0], [4, 0, 1, 1]],
             [[6, 2, 6, 2], [1, 0, 1, 0], [2, 1, 2, 1]],
             [[2, 1, 2, 1], [0, 0, 0, 0], [1, 2, 1, 2]]])
        expected1 = torch.tensor(
            [[[2, 1, 0, 2], [1, 2, 5, 0], [2, 0, 1, 1]],
             [[6, 2, 6, 2], [1, 0, 1, 0], [2, 1, 2, 1]],
             [[2, 1, 2, 1], [0, 0, 0, 0], [1, 2, 1, 2]]])
        aggregated_fbank1 = self.aggregator.aggregate_fbank_explanations(
            aggregation_indices=[(0, 0), (1, 3), (4, 4)],
            fbank_heatmap=fbank_explanation,
            include_eos=True)
        aggregated_fbank2 = self.aggregator.aggregate_fbank_explanations(
            aggregation_indices=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            fbank_heatmap=fbank_explanation,
            include_eos=True)
        self.assertTrue(torch.equal(aggregated_fbank1, expected1))
        self.assertTrue(torch.equal(aggregated_fbank2, fbank_explanation))

    def test_aggregate_tgt_explanations_rows_discrete(self):
        tgt_explanations = torch.tensor(
            [[[1.], [0.], [0.], [0.], [0.], [0.]],
             [[0.], [3.], [0.], [0.], [0.], [0.]],
             [[2.], [1.5], [1.5], [0.], [0.], [0.]],
             [[1.], [1.5], [0.], [2.], [0.], [0.]],
             [[1.], [1.], [3.5], [0.5], [6.5], [0.]],
             [[1.], [1.], [3.], [2.], [1.], [1.]]])
        aggregation_indices4 = [(1, 2), (4, 5)]
        expected4 = torch.tensor(
            [[[0.5], [0.], [0.], [0.], [0.], [0.]],
             [[1.], [1.25], [1.75], [1.25], [0.], [0.]]])
        aggregated_explanation4 = self.aggregator.aggregate_tgt_explanations_rows(
            aggregation_indices4, tgt_explanations, False)
        self.assertTrue(torch.equal(expected4, aggregated_explanation4))

    def test_aggregate_tgt_explanations_rows_continuous(self):
        tgt_explanations = torch.tensor(
            [[[1., 2., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[0., 2., 1.], [3., 0., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[2., 2., 2.], [1.5, 1.5, 1.5], [1.5, 2., 3.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[1., 1., 0.5], [1.5, 1., 1.], [0., 1., 1.], [2., 2., 2.], [0., 0., 0.], [0., 0., 0.]],
             [[1., 1.5, 2.5], [1., 1.5, 2.5], [3.5, 1.5, 2.5], [3., 0.5, 2.], [6.5, 0., 2.], [0., 0., 0.]],
             [[1., 1.5, 2.5], [3.5, 1.5, 2.5], [6.5, 0., 2.], [2., 0., 0.], [1., 0., 1.], [1., 1., 1.]]])
        aggregation_indices1 = [(1, 2), (4, 5)]
        aggregation_indices2 = [(0, 0), (1, 2), (4, 5)]
        expected1 = torch.tensor(
            [[[0.5, 2., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[1., 1.25, 1.5], [1.25, 1.25, 1.75], [1.75, 1.25, 1.75], [2.5, 1.25, 2.], [0., 0., 0.], [0., 0., 0.]]])
        expected2= torch.tensor(
            [[[0.5, 2., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[1., 1.25, 1.5], [1.25, 1.25, 1.75], [1.75, 1.25, 1.75], [2.5, 1.25, 2.], [0., 0., 0.], [0., 0., 0.]],
             [[1., 1.5, 2.5], [3.5, 1.5, 2.5], [6.5, 0., 2.], [2., 0., 0.], [1., 0., 1.], [1., 1., 1.]]])
        aggregated_explanation1 = self.aggregator.aggregate_tgt_explanations_rows(
            aggregation_indices1, tgt_explanations, False)
        aggregated_explanation2 = self.aggregator.aggregate_tgt_explanations_rows(
            aggregation_indices2, tgt_explanations, True)
        self.assertTrue(torch.equal(expected1, aggregated_explanation1))
        self.assertTrue(torch.equal(expected2, aggregated_explanation2))

    def test_call_with_no_indices(self):
        explanations = copy.deepcopy(MOCK_EXPLANATIONS)
        aggregated_explanations = self.aggregator(explanations)
        expected = {
            0: {
                "fbank_heatmap": torch.tensor(  # (4, 2, 3)
                    [[[1., 4., 1.],
                      [1., 4., 1.]],
                     [[0., 2., 0.],
                      [0., 2., 0.]],
                     [[1., 5., 1.],
                      [1., 5., 1.]],
                     [[0., 2., 0.],
                      [0., 2., 0.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (4, 4, 1)
                    [[[2.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.6], [1.], [1.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [0.]],
                     [[0.5], [1.5], [0.5], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', '▁Swim', 'm', 'ing', '▁is', '▁hard'],
                "tgt_phrases": ['</s>', 'Swimming', 'is', 'hard']},
            1: {
                "fbank_heatmap": torch.tensor(  # (6, 4, 1)
                    [[[1.], [5.], [2.], [1.]],
                     [[2.], [1.], [1.], [3.]],
                     [[1.], [1.], [1.], [0.]],
                     [[1.], [3], [1.25], [0.5]],
                     [[1.], [2.], [1.], [2.]],
                     [[2.], [1.], [2.], [1.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (6, 6, 1)
                    [[[1.], [0.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [1.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', '▁You', "'", 're', '▁co', 'rrect', '.'],
                "tgt_phrases": ['</s>', 'You', "'", 're', 'correct', '.']}}

        self.assertEqual(list(expected.keys()), list(aggregated_explanations.keys()))
        for i in aggregated_explanations:
            self.assertEqual(
                list(aggregated_explanations[i].keys()),
                ["fbank_heatmap", "tgt_embed_heatmap", "tgt_text", "tgt_phrases"])
            self.assertTrue(torch.equal(
                aggregated_explanations[i]["fbank_heatmap"], expected[i]["fbank_heatmap"]))
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["tgt_embed_heatmap"], expected[i]["tgt_embed_heatmap"]))
            self.assertEqual(aggregated_explanations[i]["tgt_text"], expected[i]["tgt_text"])
            self.assertEqual(aggregated_explanations[i]["tgt_phrases"], expected[i]["tgt_phrases"])

    def test_call_with_indices(self):
        explanations = copy.deepcopy(MOCK_EXPLANATIONS)
        indices = {0: (['Swimming', 'hard'], [(1, 3), (5, 5)]), 1: ([], [])}
        aggregated_explanations = self.aggregator(explanations, indices)
        expected = {
            0: {
                "fbank_heatmap": torch.tensor(  # (2, 2, 3)
                    [[[1., 4., 1.],
                      [1., 4., 1.]],
                     [[1., 5., 1.],
                      [1., 5., 1.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (2, 3, 1)
                    [[[2.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [0.]]]),
                "tgt_text": ['</s>', '▁Swim', 'm', 'ing', '▁is', '▁hard'],
                "tgt_phrases": ['Swimming', 'hard']}}
        self.assertEqual(list(expected.keys()), list(aggregated_explanations.keys()))
        for i in aggregated_explanations:
            self.assertEqual(
                list(aggregated_explanations[i].keys()),
                ["fbank_heatmap", "tgt_embed_heatmap", "tgt_text", "tgt_phrases"])
            self.assertTrue(torch.equal(
                aggregated_explanations[i]["fbank_heatmap"], expected[i]["fbank_heatmap"]))
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["tgt_embed_heatmap"], expected[i]["tgt_embed_heatmap"]))
            self.assertEqual(aggregated_explanations[i]["tgt_text"], expected[i]["tgt_text"])
            self.assertEqual(aggregated_explanations[i]["tgt_phrases"], expected[i]["tgt_phrases"])

    def test_call_with_indices_eos(self):
        explanations = copy.deepcopy(MOCK_EXPLANATIONS)
        indices = {0: (['</s>', 'Swimming', 'hard'], [(0, 0), (1, 3), (5, 5)]), 1: ([], [])}
        aggregated_explanations = self.aggregator(explanations, indices)
        expected = {
            0: {
                "fbank_heatmap": torch.tensor(  # (2, 2, 3)
                    [[[1., 4., 1.],
                      [1., 4., 1.]],
                     [[1., 5., 1.],
                      [1., 5., 1.]],
                     [[0., 2., 0.],
                      [0., 2., 0.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (2, 3, 1)
                    [[[2.], [0.], [0.], [0.], [0.], [0.]],
                     [[1.], [1.], [1.], [1.], [1.], [0.]],
                     [[0.5], [1.5], [0.5], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', '▁Swim', 'm', 'ing', '▁is', '▁hard'],
                "tgt_phrases": ['</s>', 'Swimming', 'hard']}}
        self.assertEqual(list(expected.keys()), list(aggregated_explanations.keys()))
        for i in aggregated_explanations:
            self.assertEqual(
                list(aggregated_explanations[i].keys()),
                ["fbank_heatmap", "tgt_embed_heatmap", "tgt_text", "tgt_phrases"])
            self.assertTrue(torch.equal(
                aggregated_explanations[i]["fbank_heatmap"], expected[i]["fbank_heatmap"]))
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["tgt_embed_heatmap"], expected[i]["tgt_embed_heatmap"]))
            self.assertEqual(aggregated_explanations[i]["tgt_text"], expected[i]["tgt_text"])
            self.assertEqual(aggregated_explanations[i]["tgt_phrases"], expected[i]["tgt_phrases"])


class TestPhraseAndWordLevelAggregator(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregator = PhraseAndWordLevelAggregator()

    def test_aggregate_tgt_explanations_cols_discrete(self):
        indices = [(0, 0), (1, 2), (3, 4), (5, 5)]
        explanation_discrete = torch.tensor(
            [[[1.], [0.], [0.], [0.], [0.], [0.]],
             [[2.], [1.], [0.], [0.], [0.], [0.]],
             [[3.], [2.], [1.], [0.], [0.], [0.]],
             [[0.], [2.], [1.], [2.], [0.], [0.]],
             [[3.], [2.], [2.], [0.], [3.], [0.]],
             [[0.], [2.], [1.], [1.], [2.], [1.]]])
        expected_discrete = torch.tensor(
            [[[1.], [0.], [0.], [0.]],
             [[2.], [0.5], [0.], [0.]],
             [[3.], [1.5], [0.], [0.]],
             [[0.], [1.5], [1], [0.]],
             [[3.], [2.], [1.5], [0.]],
             [[0.], [1.5], [1.5], [1.]]])
        aggregated_explanation = self.aggregator.aggregate_tgt_explanations_cols(
            indices, explanation_discrete)
        self.assertTrue(torch.equal(expected_discrete, aggregated_explanation))

    def test_aggregate_tgt_explanations_cols_continuous(self):
        indices = [(0, 0), (1, 2), (3, 4), (5, 5)]
        explanation_continuous = torch.tensor(
            [[[1., 1., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[2., 2., 2.], [1., 1., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[3., 3., 3.], [2., 2., 2.], [1., 1., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[0., 0., 0.], [2., 2., 2.], [1., 1., 1.], [2., 2., 2.], [0., 0., 0.], [0., 0., 0.]],
             [[3., 3., 3.], [2., 2., 2.], [2., 2., 2.], [0., 0., 0.], [3., 3., 3.], [0., 0., 0.]],
             [[0., 0., 0.], [2., 2., 2.], [1., 1., 1.], [1., 1., 1.], [2., 2., 2.], [1., 1., 1.]]])
        expected_continuous = torch.tensor(
            [[[1., 1., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
             [[2., 2., 2.], [0.5, 0.5, 0.5], [0., 0., 0.], [0., 0., 0.]],
             [[3., 3., 3.], [1.5, 1.5, 1.5], [0., 0., 0.], [0., 0., 0.]],
             [[0., 0., 0.], [1.5, 1.5, 1.5], [1., 1., 1.], [0., 0., 0.]],
             [[3., 3., 3.], [2., 2., 2.], [1.5, 1.5, 1.5], [0., 0., 0.]],
             [[0., 0., 0.], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1., 1., 1.]]])
        aggregated_explanation = self.aggregator.aggregate_tgt_explanations_cols(
            indices, explanation_continuous)
        self.assertTrue(torch.equal(expected_continuous, aggregated_explanation))

    def test_call(self):
        explanations = copy.deepcopy(MOCK_EXPLANATIONS)
        expected = {
            0: {
                "fbank_heatmap": torch.tensor(  # (4, 2, 3)
                    [[[1., 4., 1.],
                      [1., 4., 1.]],
                     [[0., 2., 0.],
                      [0., 2., 0.]],
                     [[1., 5., 1.],
                      [1., 5., 1.]],
                     [[0., 2., 0.],
                      [0., 2., 0.]]]),
                "tgt_embed_heatmap": torch.tensor(  # (4, 4, 1)
                    [[[2.], [0.], [0.], [0.]],
                     [[1.], [1.2], [0.], [0.]],
                     [[1.], [1.], [1.], [0.]],
                     [[0.5], [1.], [1.], [1.]]]),
                "tgt_text": ['</s>', 'Swimming', 'is',  'hard'],
                "tgt_phrases": ['</s>', 'Swimming', 'is', 'hard']},
            1: {
                "fbank_heatmap": torch.tensor(  # (6, 4, 1)
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
        aggregated_explanations = self.aggregator(explanations)

        self.assertEqual(list(expected.keys()), list(aggregated_explanations.keys()))
        for i in aggregated_explanations:
            self.assertEqual(
                list(aggregated_explanations[i].keys()),
                ["fbank_heatmap", "tgt_embed_heatmap", "tgt_text", "tgt_phrases"])
            self.assertTrue(torch.equal(
                aggregated_explanations[i]["fbank_heatmap"], expected[i]["fbank_heatmap"]))
            self.assertTrue(torch.allclose(
                aggregated_explanations[i]["tgt_embed_heatmap"], expected[i]["tgt_embed_heatmap"]))
            self.assertEqual(aggregated_explanations[i]["tgt_text"], expected[i]["tgt_text"])
            self.assertEqual(aggregated_explanations[i]["tgt_phrases"], expected[i]["tgt_phrases"])


if __name__ == '__main__':
    unittest.main()
