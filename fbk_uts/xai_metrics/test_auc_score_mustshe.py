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
import numpy as np
import pandas as pd

from examples.speech_to_text.xai_metrics.auc_score_mustshe import compute_score, filter_articles


class TestGenderMetricSpeechToTextExplanations(unittest.TestCase):
    def setUp(self):
        self.df_hypos = pd.DataFrame({
            "id": ["it-0002_0", "it-0002_0"],
            "audio": ["audio_path", "audio_path"],
            "n_frames": [1929, 1929],
            "src_text": ["I am pretty and turtle", "I am pretty and turtle"],
            "tgt_text": ["▁io ▁sono ▁carina ▁e ▁tartaruga", "▁io ▁sono ▁carina ▁e ▁tartaruga"],
            "speaker": ["speaker_id", "speaker_id"],
            "found_terms": ["carina", "tartaruga"],
            "found_term_pairs": ["carina carino", "tartarugo tartaruga"],
            "gender_term_indices": ["2-2", "4-4"],
            "swapped_tgt_text": ["▁io ▁sono ▁carino ▁e ▁tartaruga", "▁io ▁sono ▁carina ▁e ▁tartarugo"],
            "category": ["1F", "1M"],
            "tgt_lang": ["it", "it"]})
        
        tsv_reader_output = [{
                "id": "it-0002_0",
                "audio": "audio_path",
                "n_frames": 1929,
                "src_text": "I am pretty and turtle",
                "tgt_text": "▁io ▁sono ▁carina ▁e ▁tartaruga",
                "speaker": "speaker_id",
                "found_terms": "carina",
                "found_term_pairs": "carina carino",
                "gender_term_indices": "2-2",
                "swapped_tgt_text": "▁io ▁sono ▁carino ▁e ▁tartaruga",
                "category": "1F",
                "tgt_lang": "it"},
                {
                "id": "it-0002_0",
                "audio": "audio_path",
                "n_frames": 1929,
                "src_text": "I am pretty and turtle",
                "tgt_text": "▁io ▁sono ▁carina ▁e ▁tartaruga",
                "speaker": "speaker_id",
                "found_terms": "tartaruga",
                "found_term_pairs": "tartarugo tartaruga",
                "gender_term_indices": "4-4",
                "swapped_tgt_text": "▁io ▁sono ▁carina ▁e ▁tartarugo",
                "category": "1M",
                "tgt_lang": "it"}]
        self.steps_to_refs = {0: tsv_reader_output, 1: tsv_reader_output}
        
        self.steps_to_hypos = {
            0: ["▁io ▁sono ▁carina ▁e ▁tartaruga", "▁io ▁sono ▁carina ▁e ▁tartaruga"],
            1: ["▁io ▁sono ▁carina ▁e ▁tartaruga", "▁io ▁sono ▁carina ▁e ▁tartarugo"]}
        
        self.df_pos = pd.DataFrame({
            "ID": ["it-0002"],
            "GENDERTERMS": ["carina carino;tartarugo tartaruga"],
            "POS": ["Art/Prep;Noun"]})

    def test_filter_articles(self):
        refs_no_articles = {
            0: [{
                "id": "it-0002_0",
                "audio": "audio_path",
                "n_frames": 1929,
                "src_text": "I am pretty and turtle",
                "tgt_text": "▁io ▁sono ▁carina ▁e ▁tartaruga",
                "speaker": "speaker_id",
                "found_terms": "tartaruga",
                "found_term_pairs": "tartarugo tartaruga",
                "gender_term_indices": "4-4",
                "swapped_tgt_text": "▁io ▁sono ▁carina ▁e ▁tartarugo",
                "category": "1M",
                "tgt_lang": "it"}],
            1: [{
                "id": "it-0002_0",
                "audio": "audio_path",
                "n_frames": 1929,
                "src_text": "I am pretty and turtle",
                "tgt_text": "▁io ▁sono ▁carina ▁e ▁tartaruga",
                "speaker": "speaker_id",
                "found_terms": "tartaruga",
                "found_term_pairs": "tartarugo tartaruga",
                "gender_term_indices": "4-4",
                "swapped_tgt_text": "▁io ▁sono ▁carina ▁e ▁tartarugo",
                "category": "1M",
                "tgt_lang": "it"}]}
        hypos_no_articles = {
            0: ["▁io ▁sono ▁carina ▁e ▁tartaruga"],
            1: ["▁io ▁sono ▁carina ▁e ▁tartarugo"]}
        new_steps_to_hypos, new_steps_to_refs = filter_articles(
            self.steps_to_hypos, self.steps_to_refs, self.df_pos)
        self.assertEqual(new_steps_to_refs, refs_no_articles)
        self.assertEqual(new_steps_to_hypos, hypos_no_articles)

    def test_compute_score_gender_accuracy(self):
        scores = compute_score("gender_accuracy", self.steps_to_hypos, self.steps_to_refs)
        expected_scores = np.array([0.5, 1.])  # score for each step
        self.assertTrue(np.array_equal(scores, expected_scores))

    def test_compute_score_gender_accuracy_by_category(self):
        scores = compute_score("gender_accuracy", self.steps_to_hypos, self.steps_to_refs, categories=["1F"])
        expected_scores_1F = np.array([1., 1.])
        self.assertTrue(np.array_equal(scores, expected_scores_1F))

        scores = compute_score("gender_accuracy", self.steps_to_hypos, self.steps_to_refs, categories=["1M"])
        expected_scores_1M = np.array([0., 1.])
        self.assertTrue(np.array_equal(scores, expected_scores_1M))

    def test_compute_score_gender_coverage(self):
        scores = compute_score("gender_coverage", self.steps_to_hypos, self.steps_to_refs)
        expected_scores = np.array([1., 1.])
        self.assertTrue(np.array_equal(scores, expected_scores))

    def test_compute_score_gender_flip_rate(self):
        scores = compute_score("gender_flip_rate", self.steps_to_hypos, self.steps_to_refs)
        expected_scores = np.array([0., 0.5])
        self.assertTrue(np.array_equal(scores, expected_scores))


if __name__ == "__main__":
    unittest.main()
