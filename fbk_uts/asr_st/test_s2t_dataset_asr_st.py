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
from unittest.mock import patch

import numpy as np

from examples.speech_to_text.data.speech_to_text_dataset_asr_st import SpeechToTextDatasetASRST
from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc
from fairseq.data import Dictionary


class MockS2TDataConfigASRST(S2TDataConfigSrc):
    def __init__(self):
        self.config = {"prepend_tgt_lang_tag": True}


class ASRSTDatasetSetup:
    def setUp(self):
        self.src_dict = Dictionary(extra_special_symbols=["<lang:TGTLANG>", "<lang:SRCLANG>"])
        self.src_lines = ["I like quokkas", "I like tortoises", "I like"]
        for l in self.src_lines:
            self.src_dict.encode_line(l)
        self.tgt_dict = Dictionary(extra_special_symbols=["<lang:TGTLANG>", "<lang:SRCLANG>"])
        self.tgt_lines = [
            "Mi piacciono i quokka", "Mi piacciono le tartarughe", "Mi piacciono"]
        for l in self.tgt_lines:
            self.tgt_dict.encode_line(l)
        self.ds = SpeechToTextDatasetASRST(
            "quokka",
            True,
            MockS2TDataConfigASRST(),
            ["f1.wav", "f2.wav", "f3.wav"],
            [30, 100, 27],
            self.src_lines,
            self.tgt_lines,
            ["s1", "s2", "s3"],
            ["SRCLANG", "SRCLANG", "SRCLANG"],
            ["TGTLANG", "TGTLANG", "TGTLANG"],
            tgt_dict=self.tgt_dict,
            src_dict=self.src_dict,
        )


class SpeechASRSTDatasetTestCase(ASRSTDatasetSetup, unittest.TestCase):
    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_collater(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        self.assertEqual(samples["transcript_lengths"].tolist(), [4, 4, 3])
        self.assertEqual(samples["prepended_transcript_lengths"].tolist(), [5, 5, 4])
        self.assertEqual(samples["transcript"].tolist(), [
            [6, 7, 8, 2], [6, 7, 9, 2], [6, 7, 2, 1]])
        self.assertEqual(samples["prepended_transcript"].tolist(), [
            [5, 6, 7, 8, 2], [5, 6, 7, 9, 2], [5, 6, 7, 2, 1]])
        expected_strings = ["I like quokkas", "I like tortoises", "I like <pad>"]
        for i in range(3):
            self.assertEqual(expected_strings[i], self.src_dict.string(samples["transcript"][i]))
        expected_strings = [
            "<lang:SRCLANG> I like quokkas", "<lang:SRCLANG> I like tortoises",
            "<lang:SRCLANG> I like <pad>"]
        for i in range(3):
            self.assertEqual(expected_strings[i], self.src_dict.string(samples["prepended_transcript"][i]))

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_dataset_with_none_src_lang(self, mock_get_features_or_waveform):
        """Test dataset behavior when source language ID is None"""
        with self.assertRaises(AssertionError):
            self.ds_none_src_lang = SpeechToTextDatasetASRST(
                "quokka_none_src",
                True,
                MockS2TDataConfigASRST(),
                ["f1.wav", "f2.wav", "f3.wav"],
                [30, 100, 27],
                self.src_lines,
                self.tgt_lines,
                ["s1", "s2", "s3"],
                [None, None, None],  # Source language IDs are None
                ["TGTLANG", "TGTLANG", "TGTLANG"],
                tgt_dict=self.tgt_dict,
                src_dict=self.src_dict,
            )


if __name__ == '__main__':
    unittest.main()
