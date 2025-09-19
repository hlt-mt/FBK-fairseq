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
import pandas as pd
from unittest.mock import patch
import sentencepiece as spm

from examples.speech_to_text.scripts.xai.prep_mustshe_hyps_for_explanation import \
    filter_gender_terms, find_terms_bpe, swap_gender_hypothesis


class MockSentencePieceProcessor(spm.SentencePieceProcessor):

    def decode(self, bpe_tokenized_text):
        return ' '.join(bpe_tokenized_text).replace('▁', '')

    def encode(self, raw_text, **kwargs):
        return ['▁' + w for w in raw_text.split()]


class MustShePrepTestCase(unittest.TestCase):

    def setUp(self):
        self.sp = MockSentencePieceProcessor()

    def test_filter_gender_terms_base(self):
        row = {'moses_tgt_text': 'io sono carino', 'gender_terms': 'carino carina'}
        found_terms, found_term_pairs = filter_gender_terms(row)
        self.assertEqual(found_terms, 'carino')
        self.assertEqual(found_term_pairs, 'carino carina')

    def test_filter_gender_terms_no_terms(self):
        row = {'moses_tgt_text': 'io sono tartarugo', 'gender_terms': 'carino carina'}
        found_terms, found_term_pairs = filter_gender_terms(row)
        self.assertEqual(found_terms, '')
        self.assertEqual(found_term_pairs, '')

    def test_filter_gender_terms_multiple(self):
        row = {
            'moses_tgt_text': 'io sono carino e tartarugo', 
            'gender_terms': 'carino carina;tartaruga tartarugo'}
        found_terms, found_term_pairs = filter_gender_terms(row)
        self.assertEqual(found_terms, 'carino;tartarugo')
        self.assertEqual(found_term_pairs, 'carino carina;tartaruga tartarugo')

    def test_filter_gender_terms_duplicate(self):
        # If both the correct and wrong terms are found, we discard them from our analysis
        # since we cannot determine which one refers to the entity in question.
        row = {
            'moses_tgt_text': 'io sono carino e carina', 
            'gender_terms': 'carino carina'}
        found_terms, found_term_pairs = filter_gender_terms(row)
        self.assertEqual(found_terms, '')
        self.assertEqual(found_term_pairs, '')

    def test_find_terms_bpe_base(self):
        row = {
            'tgt_text': '▁io ▁sono ▁carino',
            'found_terms': 'carino',
            'found_term_pairs': 'carino carina'}
        gender_term_indices = find_terms_bpe(row, self.sp)
        self.assertEqual(gender_term_indices, '2-2')

    def test_find_terms_bpe_no_terms(self):
        row = {
            'tgt_text': '▁io ▁sono ▁tartarugo',
            'found_terms': 'carino',
            'found_term_pairs': 'carino carina'}
        gender_term_indices = find_terms_bpe(row, self.sp)
        self.assertEqual(gender_term_indices, '')

    def test_find_terms_bpe_multiple(self):
        row = {
            'tgt_text': '▁io ▁sono ▁carino ▁e ▁tartarugo', 
            'found_terms': 'carino;tartarugo',
            'found_term_pairs': 'carino carina;tartarugo tartaruga'}
        gender_term_indices = find_terms_bpe(row, self.sp)
        self.assertEqual(gender_term_indices, '2-2;4-4')

    def test_swap_gender_hypothesis_base(self):
        row = {
            'tgt_text': '▁io ▁sono ▁carino',
            'found_terms': 'carino',
            'found_term_pairs': 'carino carina',
            'gender_terms_indices': '2-2'}
        swapped_hyp = swap_gender_hypothesis(row, self.sp)
        self.assertEqual(swapped_hyp, '▁io ▁sono ▁carina')

    def test_swap_gender_hypothesis_no_terms(self):
        row = { 
            'tgt_text': '▁io ▁sono ▁tartarugo',
            'found_terms': '',
            'found_term_pairs': '',
            'gender_terms_indices': ''}
        swapped_hyp = swap_gender_hypothesis(row, self.sp)
        self.assertEqual(swapped_hyp, '▁io ▁sono ▁tartarugo')

    def test_swap_gender_hypothesis_multiple(self):
        row = {
            'tgt_text': '▁io ▁sono ▁carino ▁e ▁tartarugo', 
            'found_terms': 'carino;tartarugo',
            'found_term_pairs': 'carino carina;tartarugo tartaruga',
            'gender_terms_indices': '2-2;4-4'}
        swapped_hyp = swap_gender_hypothesis(row, self.sp)
        self.assertEqual(swapped_hyp, '▁io ▁sono ▁carina ▁e ▁tartaruga')


if __name__ == '__main__':
    unittest.main()
