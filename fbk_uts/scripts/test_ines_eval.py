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
import tempfile

from examples.speech_to_text.scripts.gender.INES_eval import sentence_level_scores, global_inclusivity_index, global_accuracy


class InesEvalTestCase(unittest.TestCase):

    def test_sentence_level_scores_base_not_inclusive(self):
        hypos = ["The average man spends about eight hours a day with sleep ."]
        # Create temporary files for object acceptable to function in INES_eval
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
        self.assertDictEqual(out[0], {
            "num_terms_found": 1,
            "num_inclusive": 0,
            "num_not_inclusive": 1})

    def test_sentence_level_scores_base_inclusive(self):
        # mind the \ I missed in my previous comment
        hypos = ["The average person spends about eight hours a day with sleep ."]
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
            self.assertDictEqual(out[0], {
                "num_terms_found": 1,
                "num_inclusive": 1,
                "num_not_inclusive": 0})

    def test_sentence_level_scores_partial_match(self):
        hypos = ["I am the average male ."]
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
        self.assertDictEqual(out[0], {
            "num_terms_found": 0,
            "num_inclusive": 0,
            "num_not_inclusive": 0})

    def test_sentence_level_scores_inconsecutive_tokens(self):
        hypos = ["I am the average male , while you are a good person ."]
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
        self.assertDictEqual(out[0], {
            "num_terms_found": 0,
            "num_inclusive": 0,
            "num_not_inclusive": 0})

    def test_sentence_level_scores_consecutive_tokens(self):
        hypos = ["The average dog for the average man ."]
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
        self.assertDictEqual(out[0], {
            "num_terms_found": 1,
            "num_inclusive": 0,
            "num_not_inclusive": 1})

    def test_sentence_level_scores_both_term_in(self):
        hypos = ["The average person is an average man ."]
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
        self.assertDictEqual(out[0], {
            "num_terms_found": 1,
            "num_inclusive": 1,
            "num_not_inclusive": 1})

    def test_sentence_level_scores_both_term_ni(self):
        hypos = ["The average man is an average person ."]
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
        self.assertDictEqual(out[0], {
            "num_terms_found": 1,
            "num_inclusive": 1,
            "num_not_inclusive": 1})

    def test_sentence_level_scores_duplicate_term(self):
        hypos = ["The average person is an average person ."]
        with tempfile.NamedTemporaryFile(mode="w") as hypos_file, \
                tempfile.NamedTemporaryFile(mode="w") as tsv_file:
            # here write the hypos to hypos_file and the tsv_file
            hypos_file.write('\n'.join(hypos))
            tsv_file.write("ID\tEVAL-TERMS-en\n1\taverage person;average man")
            hypos_file.flush()
            tsv_file.flush()
            out = sentence_level_scores(hypos_file.name, tsv_file.name)
        self.assertDictEqual(out[0], {
            "num_terms_found": 1,
            "num_inclusive": 1,
            "num_not_inclusive": 0})

    def test_global_inclusivity_index_empty(self):
        with self.assertRaises(Exception) as e:
            out = global_inclusivity_index([])
        self.assertEqual(str(e.exception), "Cannot evaluate with empty INES TSV")

    def test_global_accuracy_empty(self):
        with self.assertRaises(Exception) as e:
            out = global_accuracy([])
        self.assertEqual(str(e.exception), "Cannot evaluate with empty INES TSV")

    def test_global_accuracy(self):
        sentence_scores = [
            {"num_terms_found": 1, "num_inclusive": 0, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 0},
            {"num_terms_found": 0, "num_inclusive": 0, "num_not_inclusive": 0},
            {"num_terms_found": 0, "num_inclusive": 0, "num_not_inclusive": 0},
            {"num_terms_found": 1, "num_inclusive": 0, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 0}]
        global_score = global_accuracy(sentence_scores)
        self.assertEqual(global_score.term_coverage, 0.75)
        self.assertEqual(global_score.inclusivity_accuracy, 0.5)

    def test_inclusivity_index(self):
        sentence_scores = [
            {"num_terms_found": 1, "num_inclusive": 0, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 0},
            {"num_terms_found": 0, "num_inclusive": 0, "num_not_inclusive": 0},
            {"num_terms_found": 0, "num_inclusive": 0, "num_not_inclusive": 0},
            {"num_terms_found": 1, "num_inclusive": 0, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 1},
            {"num_terms_found": 1, "num_inclusive": 1, "num_not_inclusive": 0}]
        global_score = global_inclusivity_index(sentence_scores)
        self.assertEqual(global_score, 0.5)


if __name__ == '__main__':
    unittest.main()
