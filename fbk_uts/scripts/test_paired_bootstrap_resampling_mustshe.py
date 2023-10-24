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
import io
from contextlib import redirect_stdout
import unittest

from examples.speech_to_text.scripts.gender import \
    mustshe_gender_accuracy as gender_acc, \
    paired_bootstrap_resampling_mustshe as pb_mustshe


class TestPairedBootstrapMustshe(unittest.TestCase):
    def setUp(self):
        # Create a list of instances of the named tuple defined in mustshe_gender_accuracy
        self.mustshe_entries = [
            gender_acc.MuSTSheEntry(
                category="1M", genderterms=[
                    gender_acc.GenderTerms(correct="stato", wrong="stata"),
                    gender_acc.GenderTerms(correct="un", wrong="una")]),
            gender_acc.MuSTSheEntry(
                category="1M", genderterms=[
                    gender_acc.GenderTerms(correct="contento", wrong="contenta")]),
            gender_acc.MuSTSheEntry(
                category="2F", genderterms=[
                    gender_acc.GenderTerms(correct="arrivata", wrong="arrivato")]),
            gender_acc.MuSTSheEntry(
                category="2F", genderterms=[
                    gender_acc.GenderTerms(correct="pensata", wrong="pensato")])]
        self.stats_baseline = [
            {'num_terms': 2, 'num_terms_found': 2, 'num_correct': 1, 'num_wrong': 1},  # 1M
            {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 1, 'num_wrong': 0},  # 1M
            {'num_terms': 1, 'num_terms_found': 0, 'num_correct': 0, 'num_wrong': 0},  # 2F
            {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 0, 'num_wrong': 1}]  # 2F
        self.stats_experimental1 = [
            {'num_terms': 2, 'num_terms_found': 2, 'num_correct': 2, 'num_wrong': 1},  # 1M
            {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 1, 'num_wrong': 0},  # 1M
            {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 0, 'num_wrong': 1},  # 2F
            {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 1, 'num_wrong': 0}]  # 2F
        # statistics of a new experimental model which is identical to the baseline
        self.stats_experimental2 = [
            {'num_terms': 2, 'num_terms_found': 2, 'num_correct': 1, 'num_wrong': 1},  # 1M
            {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 1, 'num_wrong': 0},  # 1M
            {'num_terms': 1, 'num_terms_found': 0, 'num_correct': 0, 'num_wrong': 0},  # 2F
            {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 0, 'num_wrong': 1}]  # 2F

    def test_filter_by_category(self):
        expected_result = {
            "1M": {
                "baseline_stats": [
                    {'num_terms': 2, 'num_terms_found': 2, 'num_correct': 1, 'num_wrong': 1},
                    {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 1, 'num_wrong': 0}],
                "experimental_stats": [
                    {'num_terms': 2, 'num_terms_found': 2, 'num_correct': 2, 'num_wrong': 1},
                    {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 1, 'num_wrong': 0}],
                "mustshe_entries": [
                    gender_acc.MuSTSheEntry(
                        category="1M",
                        genderterms=[
                            gender_acc.GenderTerms(correct="stato", wrong="stata"),
                            gender_acc.GenderTerms(correct="un", wrong="una")]),
                    gender_acc.MuSTSheEntry(
                        category="1M", genderterms=[
                            gender_acc.GenderTerms(correct="contento", wrong="contenta")])]},
            "2F": {
                "baseline_stats": [
                    {'num_terms': 1, 'num_terms_found': 0, 'num_correct': 0, 'num_wrong': 0},
                    {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 0, 'num_wrong': 1}],
                "experimental_stats": [
                    {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 0, 'num_wrong': 1},
                    {'num_terms': 1, 'num_terms_found': 1, 'num_correct': 1, 'num_wrong': 0}],
                "mustshe_entries": [
                    gender_acc.MuSTSheEntry(
                        category="2F", genderterms=[
                            gender_acc.GenderTerms(correct="arrivata", wrong="arrivato")]),
                    gender_acc.MuSTSheEntry(
                        category="2F", genderterms=[
                            gender_acc.GenderTerms(correct="pensata", wrong="pensato")])]}}
        result = pb_mustshe.group_by_category(
            ["1M", "2F"], self.stats_baseline, self.stats_experimental1, self.mustshe_entries)
        self.assertEqual(expected_result, result)

    def test_paired_bootstrap_resample(self):
        filtered = pb_mustshe.group_by_category(
            ["1M", "2F", "Global"], self.stats_baseline, self.stats_experimental1, self.mustshe_entries)
        # Test 1
        # With seed=3, baseline_stats=4, num_samples=2, sample_size=2
        # indices = np.random.randint(low=0, high=len(baseline_stats), size=(num_samples, sample_size))
        # gives [[2 0], [1 3]]
        #
        # baseline_better_cov, experimental_better_cov, baseline_better_acc, experimental_better_acc for cat 1M
        expected_results_1M = (0, 0, 0, 1)
        results_1M = pb_mustshe.paired_bootstrap_resample(
            filtered["1M"]["baseline_stats"],
            filtered["1M"]["experimental_stats"],
            filtered["1M"]["mustshe_entries"],
            num_samples=2,
            sample_size=2,
            seed=3)
        self.assertEqual(expected_results_1M, results_1M)
        # Test 2
        # baseline_better_cov, experimental_better_cov, baseline_better_acc, experimental_better_acc for cat Global
        expected_results_Global = (0, 1, 0, 1)
        results_Global = pb_mustshe.paired_bootstrap_resample(
            filtered["Global"]["baseline_stats"],
            filtered["Global"]["experimental_stats"],
            filtered["Global"]["mustshe_entries"],
            num_samples=2,
            sample_size=2,
            seed=3)
        self.assertEqual(expected_results_Global, results_Global)

    # Check whether the output of get_print() is correct
    def test_do_print(self):
        results = {
            "1M":
                {"baseline_all_cov": 0.75,
                 "experimental_all_cov": 0.22,
                 "baseline_all_acc": 0.65,
                 "experimental_all_acc": 0.32,
                 "baseline_better_cov": 0,
                 "experimental_better_cov": 1,
                 "baseline_better_acc": 4,
                 "experimental_better_acc": 14},
            "2F":
                {"baseline_all_cov": 0.67,
                 "experimental_all_cov": 0.40,
                 "baseline_all_acc": 0.55,
                 "experimental_all_acc": 0.23,
                 "baseline_better_cov": 10,
                 "experimental_better_cov": 0,
                 "baseline_better_acc": 5,
                 "experimental_better_acc": 1}}
        content = """\
------------------------------+----------+---------------+---------------
SYSTEM                        |CATEGORY  |TERM COVERAGE  |GENDER ACCURACY
------------------------------+----------+---------------+---------------
Baseline                      |1M        |75.00          |65.00          
Experimental1234567890123456  |1M        |22.00 p=0.1    |32.00 p=1.4*   
------------------------------+----------+---------------+---------------
Baseline                      |2F        |67.00          |55.00          
Experimental1234567890123456  |2F        |40.00 p=1.0*   |23.00 p=0.5    
------------------------------+----------+---------------+---------------
"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            pb_mustshe.do_print("Baseline", "Experimental123456789012345678901234567890", results, 10, 0.05)
        self.assertEqual(content, buf.getvalue())

    # Test that the seed works
    def test_seed(self):
        filtered = pb_mustshe.group_by_category(
            ["1M", "2F", "Global"], self.stats_baseline, self.stats_experimental1, self.mustshe_entries)
        results_1 = pb_mustshe.paired_bootstrap_resample(
            filtered["Global"]["baseline_stats"],
            filtered["Global"]["experimental_stats"],
            filtered["Global"]["mustshe_entries"],
            num_samples=2,
            sample_size=2,
            seed=3)
        results_2 = pb_mustshe.paired_bootstrap_resample(
            filtered["Global"]["baseline_stats"],
            filtered["Global"]["experimental_stats"],
            filtered["Global"]["mustshe_entries"],
            num_samples=2,
            sample_size=2,
            seed=3)
        self.assertEqual(results_1, results_2)

    def test_calculate_p(self):
        _, ast1 = pb_mustshe._calculate_p(better_count=2, num_samples=4, sign_level=0.05)
        _, ast2 = pb_mustshe._calculate_p(better_count=4, num_samples=4, sign_level=0.05)
        self.assertEqual(ast1, '')
        self.assertEqual(ast2, '*')

    # Test that when there are no valid categories an Error is raised during printing
    def test_categories_print_stats(self):
        results = {}
        self.assertRaises(
            AssertionError,
            pb_mustshe.do_print,
            'Baseline',
            'Experimental',
            results,
            2,
            0.05)

    # Test critical cases where there is a high number of tied cases
    def test_tied_paired_bootstrap_resample(self):
        expected_comparison_stats = (0, 0, 0, 0)
        filtered_1M = pb_mustshe.group_by_category(
            ["1M", "2F"], self.stats_baseline, self.stats_experimental2, self.mustshe_entries)
        # baseline_better_cov, experimental_better_cov, baseline_better_acc, experimental_better_acc for cat Global
        comparison_stats = pb_mustshe.paired_bootstrap_resample(
            filtered_1M["1M"]["baseline_stats"],
            filtered_1M["1M"]["experimental_stats"],
            filtered_1M["1M"]["mustshe_entries"],
            num_samples=2,
            sample_size=2,
            seed=3)
        self.assertEqual(expected_comparison_stats, comparison_stats)
        # Check p value
        p, ast = pb_mustshe._calculate_p(
            better_count=comparison_stats[0],
            num_samples=2,
            sign_level=0.05)
        self.assertEqual((0.0, ''), (p, ast))


if __name__ == '__main__':
    unittest.main()
