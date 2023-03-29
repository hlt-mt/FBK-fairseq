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
from unittest.mock import patch

import numpy as np

from examples.speech_to_text.data.speech_to_text_dataset_aux_classification \
    import SpeechToTextDatasetAuxiliaryClassification, S2TDataConfigAuxiliaryClassification
from fairseq.data import Dictionary


class MockS2TDataConfigAuxiliaryClassification(S2TDataConfigAuxiliaryClassification):
    def __init__(self, aux_classes):
        self.config = {}
        self.__aux_classes = aux_classes

    @property
    def aux_classes(self):
        return self.__aux_classes


class AuxiliaryClassificationDatasetSetup:
    def setUp(self):
        self.src_dict = Dictionary()
        src_lines = ["I like quokkas", "I like tortoises", "I like elephants"]
        for l in src_lines:
            self.src_dict.encode_line(l)
        self.tgt_dict = Dictionary()
        tgt_lines = [
            "Mi piacciono i quokka", "Mi piacciono le tartarughe", "Mi piacciono gli elefanti"]
        for l in tgt_lines:
            self.tgt_dict.encode_line(l)
        self.ds = SpeechToTextDatasetAuxiliaryClassification(
            "quokka",
            True,
            MockS2TDataConfigAuxiliaryClassification(["F", "M"]),
            ["f1.wav", "f2.wav", "f3.wav"],
            [30, 100, 27],
            src_lines,
            tgt_lines,
            ["M", "M", "F"],
            ["s1", "s2", "s3"],
            tgt_dict=self.tgt_dict,
            src_dict=self.src_dict,
        )


class SpeechAuxiliaryClassificationDatasetTestCase(AuxiliaryClassificationDatasetSetup, unittest.TestCase):
    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_basic_usage(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([])
        self.assertEqual(self.ds[0][4], 1)
        self.assertEqual(self.ds[1][4], 1)
        self.assertEqual(self.ds[2][4], 0)
        self.assertEqual(self.tgt_dict.string(self.ds[0][2]), "Mi piacciono i quokka")
        self.assertEqual(self.src_dict.string(self.ds[0][3]), "I like quokkas")

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_collater(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        self.assertEqual([1, 1, 0], samples["auxiliary_target"].tolist())
        expected_strings = ["Mi piacciono i quokka", "Mi piacciono le tartarughe", "Mi piacciono gli elefanti"]
        for i in range(3):
            self.assertEqual(expected_strings[i], self.tgt_dict.string(samples["target"][i]))
        expected_strings = ["I like quokkas", "I like tortoises", "I like elephants"]
        for i in range(3):
            self.assertEqual(expected_strings[i], self.src_dict.string(samples["transcript"][i]))


if __name__ == '__main__':
    unittest.main()
