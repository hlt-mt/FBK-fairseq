# Copyright 2023 FBK

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

from examples.speech_to_text.scripts.gender.mustshe_gender_accuracy import MuSTSheEntry, GenderTerms, \
    sentence_level_statistics, global_scores


class GenderAccuracyTestCase(unittest.TestCase):
    def setUp(self):
        self.mustshe_def = [
            MuSTSheEntry("1F", [GenderTerms("bella", "bello"), GenderTerms("carina", "carino")]),
            MuSTSheEntry("1F", [GenderTerms("tartaruga", "tartarugo")]),
            MuSTSheEntry("2M", [GenderTerms("il", "la"), GenderTerms("tartarugo", "tartaruga")]),
        ]

    def test_sentence_stats_base(self):
        hypos = ["io sono bella e carino", "io sono tartaruga", "vedo un quokka"]
        out = sentence_level_statistics(hypos, self.mustshe_def)
        self.assertDictEqual(out[0], {
            "num_terms": 2,
            "num_terms_found": 2,
            "num_correct": 1,
            "num_wrong": 1})
        self.assertDictEqual(out[1], {
            "num_terms": 1,
            "num_terms_found": 1,
            "num_correct": 1,
            "num_wrong": 0})
        self.assertDictEqual(out[2], {
            "num_terms": 2,
            "num_terms_found": 0,
            "num_correct": 0,
            "num_wrong": 0})

    def test_sentence_stats_duplicate(self):
        hypos = ["io sono bella bello e carina", "io sono tartaruga", "vedo un quokka"]
        out = sentence_level_statistics(hypos, self.mustshe_def)
        self.assertDictEqual(out[0], {
            "num_terms": 2,
            "num_terms_found": 2,
            "num_correct": 2,
            "num_wrong": 1})

    def test_duplicate_term_in_sentence(self):
        mustshe_def = [
            MuSTSheEntry("1F", [GenderTerms("bella", "bello"), GenderTerms("bella", "bello")]),
        ]
        hypos1 = ["io sono bella e bello"]
        hypos2 = ["io sono bella e carino"]
        hypos3 = ["io sono bella e bella"]
        out1 = sentence_level_statistics(hypos1, mustshe_def)
        self.assertDictEqual(out1[0], {
            "num_terms": 2,
            "num_terms_found": 1,
            "num_correct": 1,
            "num_wrong": 1})
        out2 = sentence_level_statistics(hypos2, mustshe_def)
        self.assertDictEqual(out2[0], {
            "num_terms": 2,
            "num_terms_found": 1,
            "num_correct": 1,
            "num_wrong": 0})
        out3 = sentence_level_statistics(hypos3, mustshe_def)
        self.assertDictEqual(out3[0], {
            "num_terms": 2,
            "num_terms_found": 2,
            "num_correct": 2,
            "num_wrong": 0})

    def test_sentence_stats_empy(self):
        out = sentence_level_statistics([], [])
        self.assertTrue(len(out) == 0)

    def test_global_scores_base(self):
        sentence_scores = [
            {"num_terms": 2, "num_terms_found": 2, "num_correct": 1, "num_wrong": 1},
            {"num_terms": 1, "num_terms_found": 1, "num_correct": 1, "num_wrong": 0},
            {"num_terms": 2, "num_terms_found": 0, "num_correct": 0, "num_wrong": 0}]
        out = global_scores(sentence_scores, self.mustshe_def)
        self.assertDictEqual(
            out['1F']._asdict(), {"term_coverage": 1.0, "gender_accuracy": 2 / 3})
        self.assertDictEqual(
            out['2M']._asdict(), {"term_coverage": 0.0, "gender_accuracy": 0.0})
        self.assertDictEqual(
            out['Global']._asdict(), {"term_coverage": 3 / 5, "gender_accuracy": 2 / 3})
        # Test with coverage not 100
        sentence_scores = [
            {"num_terms": 2, "num_terms_found": 1, "num_correct": 0, "num_wrong": 1},
            {"num_terms": 1, "num_terms_found": 1, "num_correct": 1, "num_wrong": 0},
            {"num_terms": 2, "num_terms_found": 0, "num_correct": 0, "num_wrong": 0}]
        out = global_scores(sentence_scores, self.mustshe_def)
        self.assertDictEqual(
            out['1F']._asdict(), {"term_coverage": 2 / 3, "gender_accuracy": 0.5})
        self.assertDictEqual(
            out['2M']._asdict(), {"term_coverage": 0.0, "gender_accuracy": 0.0})
        self.assertDictEqual(
            out['Global']._asdict(), {"term_coverage": 2 / 5, "gender_accuracy": 0.5})

    def test_global_scores_duplicate(self):
        sentence_scores = [
            {"num_terms": 2, "num_terms_found": 2, "num_correct": 1, "num_wrong": 1},
            {"num_terms": 1, "num_terms_found": 1, "num_correct": 1, "num_wrong": 0},
            {"num_terms": 2, "num_terms_found": 2, "num_correct": 2, "num_wrong": 1}]
        out = global_scores(sentence_scores, self.mustshe_def)
        self.assertDictEqual(
            out['1F']._asdict(), {"term_coverage": 1.0, "gender_accuracy": 2 / 3})
        self.assertDictEqual(
            out['2M']._asdict(), {"term_coverage": 1.0, "gender_accuracy": 2 / 3})
        self.assertDictEqual(
            out['Global']._asdict(), {"term_coverage": 1.0, "gender_accuracy": 2 / 3})

    def test_global_scores_empty(self):
        out = global_scores([], [])
        self.assertDictEqual(out, {})


if __name__ == '__main__':
    unittest.main()
