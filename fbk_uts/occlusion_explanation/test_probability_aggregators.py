# Copyright 2025 FBK

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

from torch import LongTensor, Tensor
from examples.speech_to_text.occlusion_explanation.scorers.probability_aggregators import (
    ChainProbabilityAggregator,
    FirstOnlyProbabilityAggregator,
    LengthNormProbabilityAggregator,
    WordBoundaryProbabilityAggregator)
from fairseq.data.dictionary import Dictionary


class TestChainProbabilityAggregator(unittest.TestCase):
    def setUp(self):
        self.prob_aggregator = ChainProbabilityAggregator()

    def test_compute_word_probability(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([0])
        end_indices = LongTensor([1])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.08]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_batch(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]],
            [[0.5, 0.5, 0.2, 0.1, 0.2, 0.5],
             [0.4, 0.1, 0.2, 0.3, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1], [0, 1]])
        start_indices = LongTensor([0, 0])
        end_indices = LongTensor([1, 1])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.08]], [[0.05]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_single_token(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([0])
        end_indices = LongTensor([0])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.2]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_out_of_bounds(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([1])
        end_indices = LongTensor([2])
        with self.assertRaises(AssertionError):
            self.prob_aggregator.compute_word_probability(
                prob_distribution, tgt_tokens, start_indices, end_indices)


class TestLengthNormProbabilityAggregator(unittest.TestCase):
    def setUp(self):
        self.prob_aggregator = LengthNormProbabilityAggregator()

    def test_compute_word_probability(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([0])
        end_indices = LongTensor([1])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.282842]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_batch(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]],
            [[0.5, 0.5, 0.2, 0.1, 0.2, 0.5],
             [0.4, 0.1, 0.2, 0.3, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1], [0, 1]])
        start_indices = LongTensor([0, 0])
        end_indices = LongTensor([1, 1])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.282842]], [[0.223606]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_single_token(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([0])
        end_indices = LongTensor([0])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.2]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_out_of_bounds(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([1])
        end_indices = LongTensor([2])
        with self.assertRaises(AssertionError):
            self.prob_aggregator.compute_word_probability(
                prob_distribution, tgt_tokens, start_indices, end_indices)


class TestFirstOnlyProbabilityAggregator(unittest.TestCase):
    def setUp(self):
        self.prob_aggregator = FirstOnlyProbabilityAggregator()

    def test_compute_word_probability(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([0])
        end_indices = LongTensor([1])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.2]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_batch(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]],
            [[0.5, 0.5, 0.2, 0.1, 0.2, 0.5],
             [0.4, 0.1, 0.2, 0.3, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1], [0, 1]])
        start_indices = LongTensor([0, 0])
        end_indices = LongTensor([1, 1])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.2]], [[0.5]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_single_token(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([0])
        end_indices = LongTensor([0])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.2]]])
        self.assertTrue(word_probs.allclose(expected_probs))

    def test_compute_word_probability_out_of_bounds(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[2, 1]])
        start_indices = LongTensor([2])
        end_indices = LongTensor([2])
        with self.assertRaises(AssertionError):
            self.prob_aggregator.compute_word_probability(
                prob_distribution, tgt_tokens, start_indices, end_indices)


class TestWordBoundaryProbabilityAggregator(unittest.TestCase):
    def setUp(self):
        tgt_dict = Dictionary()
        tgt_dict.add_symbol("\u2581" + "a")
        tgt_dict.add_symbol("b")
        # tgt_dict: {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '▁a': 4, 'b': 5}
        self.prob_aggregator = WordBoundaryProbabilityAggregator(tgt_dict)

    def test_get_bow_probs(self):
        prob_distribution = Tensor(
            [[[0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
             [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]],
            [[0.5, 0.5, 0.2, 0.1, 0.2, 0.5],
             [0.4, 0.1, 0.2, 0.3, 0.4, 0.1]]])
        bow_probs = self.prob_aggregator._get_bow_probs(prob_distribution, LongTensor([0, 1]))
        expected_bow_probs = Tensor([[[0.9]], [[1.4]]])
        self.assertTrue(bow_probs.allclose(expected_bow_probs))

    def test_compute_word_probability(self):
        prob_distribution = Tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]]])
        tgt_tokens = LongTensor([[0, 3, 5, 2]])
        start_indices = LongTensor([1])
        end_indices = LongTensor([2])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.0055]]])
        self.assertTrue(word_probs.allclose(expected_probs, atol=0.0001))

    def test_compute_word_probability_batch(self):
        prob_distribution = Tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]],
            [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]]])
        tgt_tokens = LongTensor([[0, 3, 5, 2], [0, 3, 5, 2]])
        start_indices = LongTensor([1, 0])
        end_indices = LongTensor([2, 0])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.0055]], [[0.2]]])
        self.assertTrue(word_probs.allclose(expected_probs, atol=0.0001))

    def test_compute_word_probability_single_token(self):
        prob_distribution = Tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]]])
        tgt_tokens = LongTensor([[0, 3, 5, 2]])
        start_indices = LongTensor([2])
        end_indices = LongTensor([2])
        word_probs = self.prob_aggregator.compute_word_probability(
            prob_distribution, tgt_tokens, start_indices, end_indices)
        expected_probs = Tensor([[[0.025]]])
        self.assertTrue(word_probs.allclose(expected_probs, atol=0.0001))

    def test_compute_word_probability_out_of_bounds(self):
        prob_distribution = Tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[0, 3, 5, 2]])
        start_indices = LongTensor([3])
        end_indices = LongTensor([3])
        with self.assertRaises(AssertionError):
            self.prob_aggregator.compute_word_probability(
                prob_distribution, tgt_tokens, start_indices, end_indices)

    def test_compute_word_probability_last_token(self):
        prob_distribution = Tensor(
            [[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.1, 0.2, 0.1],
            [0.4, 0.4, 0.2, 0.6, 0.4, 0.1]]])
        tgt_tokens = LongTensor([[0, 3, 5]])
        start_indices = LongTensor([2])
        end_indices = LongTensor([2])
        with self.assertRaises(AssertionError):
            self.prob_aggregator.compute_word_probability(
                prob_distribution, tgt_tokens, start_indices, end_indices)


if __name__ == '__main__':
    unittest.main()
