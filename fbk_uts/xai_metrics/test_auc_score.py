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

from examples.speech_to_text.xai_metrics.auc_score import read_hypos, filter_hypos
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
        self.refs = [
            "All along, all fantasy books have always had maps, but these maps have been static.",
            "I spend most of my time in jails, in prisons, on death row."]
        self.hypos = [
            {
                0: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                1: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                2: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                3: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."]},
            {
                0: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                1: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                2: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                3: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."]},
            {
                0: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                1: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                2: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."],
                3: [
                    "All along the fantasy books that have always had maps, these maps have been static.",
                    "I spent most of the time in jails, in prisons, on that row."]}]

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

    def test_filter_hypos(self):
        for file_path, text in self.out:
            step_to_sent = filter_hypos(text, 4, 2, file_path)
            for i in range(4):
                self.assertTrue(len(step_to_sent[i]) == 2)
            self.assertFalse(4 in step_to_sent)

    def test_filter_hypos_exceptions(self):
        # test step error
        with self.assertRaises(KeyError) as error_message:
            filter_hypos(self.out[0][1], 5, 2, self.out[0][0])
        self.assertEqual(
            str(error_message.exception), f"'Step 4 not available in file {self.out[0][0]}.'")
        # test sent_id error
        with self.assertRaises(KeyError) as error_message:
            filter_hypos(self.out[0][1], 4, 3, self.out[0][0])
        self.assertEqual(
            str(error_message.exception), f"'Sentence 2 for step 0 not available in file {self.out[0][0]}.'")


if __name__ == '__main__':
    unittest.main()
