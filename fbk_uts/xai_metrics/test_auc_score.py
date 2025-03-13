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
import os
import unittest
from unittest.mock import patch, MagicMock

from examples.speech_to_text.xai_metrics.auc_score import read_hypos, filter_samples, compute_score, check_steps
from fairseq.scoring import build_scorer


class TestMetricSpeechToTextDataset(unittest.TestCase):
    def setUp(self):
        self.sacrebleu_scorer = build_scorer("sacrebleu", tgt_dict=None)
        self.current_directory = os.path.dirname(__file__)
        relative_file_path = os.path.join('mock_data', 'out')
        self.out_file_path = os.path.join(self.current_directory, relative_file_path)
        relative_folder_path = os.path.join('mock_data', 'out_folder')
        self.out_folder_path = os.path.join(self.current_directory, relative_folder_path)
        self.out = read_hypos(self.out_folder_path)
        self.tsv_data = [
            {
                "n_frames": 50,
                "tgt_text": "I spend most of my time in jails, in prisons, on death row."},
            {
                "n_frames": 100,
                "tgt_text": "All along, all fantasy books have always had maps, but these maps have been static"}]

    # test read_hypos() with hypo_path being a single output file
    def test_read_hypos_single(self):
        out = read_hypos(self.out_file_path)
        self.assertTrue(isinstance(out, list))
        self.assertTrue(len(out) == 1)

    # test read_hypos() with hypo_path being a folder containing 4 output files
    def test_read_hypos_multiple(self):
        self.assertTrue(isinstance(self.out, list))
        self.assertTrue(len(self.out) == 4)
        self.assertTrue(self.out[0] != self.out[1] != self.out[2] != self.out[3])

    # test filter_samples() with bins that include all sentences
    def test_filter_samples_all_included(self):
        for file_path, text in self.out:
            step_to_sent, filtered_refs = filter_samples(
                text, self.tsv_data, 0, 120)
            for i in range(4):
                self.assertTrue(len(step_to_sent[i]) == 2)
                self.assertTrue(len(filtered_refs[i]) == 2)
            self.assertFalse(4 in step_to_sent)
            self.assertFalse(4 in filtered_refs)

    # test filter_samples() with bins that don't include all sentences
    def test_filter_samples(self):
        for file_path, text in self.out:
            step_to_sent, filtered_refs = filter_samples(
                text, self.tsv_data, 0, 60)
            for i in range(4):
                self.assertTrue(len(step_to_sent[i]) == 1)
                self.assertTrue(len(filtered_refs[i]) == 1)

    def test_compute_score(self):
        # Create a mock scorer that has the expected methods
        with patch('examples.speech_to_text.xai_metrics.auc_score.build_scorer') as mock_build_scorer:
            mock_scorer = MagicMock()
            mock_build_scorer.return_value = mock_scorer
            mock_scorer.score.return_value = 1.0

            hypos = {
                2: ["hypo1", "hypo2", "hypo3"],
                0: ["hypo1", "hypo2", "hypo3"],
                1: ["hypo1", "hypo2", "hypo3"]}
            refs = {
                0: ["ref1", "ref2", "ref3"],
                2: ["ref1", "ref2", "ref3"],
                1: ["ref1", "ref2", "ref3"]}

            result = compute_score("dummy", refs, hypos)
            self.assertEqual(result, [1.0, 1.0, 1.0])

    # Test if the assertion error in compute_score() is raised
    def test_compute_score_assertion(self):
        # Create a mock scorer that has the expected methods
        with patch('examples.speech_to_text.xai_metrics.auc_score.build_scorer') as mock_build_scorer:
            mock_scorer = MagicMock()
            mock_build_scorer.return_value = mock_scorer
            mock_scorer.score.return_value = 1.0

            hypos = {
                2: ["hypo1", "hypo2", "hypo3"],
                0: ["hypo1", "hypo2", "hypo3"],
                1: ["hypo1", "hypo2", "hypo3"]}
            refs = {
                0: ["ref1", "ref2", "ref3"],
                2: ["ref1", "ref2", "ref3"],
                1: ["ref1", "ref2"]}  # different length for step 1

            with self.assertRaises(AssertionError):
                compute_score("dummy", refs, hypos)

    # Test if the ValueError is raised for missing steps
    def test_check_steps_missing(self):
        expected_steps = {0, 1, 2}
        step_to_hypos = {
            0: ["hypo1", "hypo2"],
            2: ["hypo1", "hypo2"]}
        file_path = "test_file.txt"

        with self.assertRaises(ValueError) as context:
            check_steps(file_path, step_to_hypos, expected_steps)
        # Check if the error message contains the missing steps
        self.assertIn("Missing steps for file test_file.txt: 1", str(context.exception))

    # Test case where there are extra steps
    def test_check_steps_extra(self):
        expected_steps = {0, 1, 2}
        step_to_hypos = {
            0: ["hypo1", "hypo2"],
            1: ["hypo1", "hypo2"],
            2: ["hypo1", "hypo2"],
            3: ["hypo1", "hypo2"]}  # Extra step
        file_path = "test_file.txt"

        with patch('examples.speech_to_text.xai_metrics.auc_score.LOGGER.warning') as mock_warning:
            check_steps(file_path, step_to_hypos, expected_steps)
            # Check if the logger output contains the extra step warning
            mock_warning.assert_called_with("Extra steps found for file test_file.txt: 3")

    # Test case where there are no missing or extra steps
    def test_check_steps_no_issue(self):
        expected_steps = {0, 1, 2}
        step_to_hypos = {
            0: ["hypo1", "hypo2"],
            1: ["hypo1", "hypo2"],
            2: ["hypo1", "hypo2"]}
        file_path = "test_file.txt"

        with patch('builtins.print') as mock_print:
            check_steps(file_path, step_to_hypos, expected_steps)
            # Ensure that print was not called
            mock_print.assert_not_called()


if __name__ == '__main__':
    unittest.main()
