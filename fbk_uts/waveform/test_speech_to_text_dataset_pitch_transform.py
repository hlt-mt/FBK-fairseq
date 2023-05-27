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
import os
import unittest
from unittest.mock import patch, mock_open

import numpy as np

from fairseq.data import Dictionary
from fairseq.data.audio.audio_utils import get_waveform
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig, SpeechToTextDataset


class MockS2TDataConfig(S2TDataConfig):
    def __init__(self, p_manipulation):
        self.config = {}
        self.p = p_manipulation

    def is_input_waveform(self):
        return True

    def waveform_sample_rate(self):
        return 44100

    def get_waveform_transforms(self, split, is_train):
        return {
            "transforms": ["random_pitch"],
            "random_pitch": {"p": self.p, "gender_tsv": "fake.tsv", "sampling_rate": 44100}}


class SpeechToTextDatasetTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # this imports and register models, tasks, and so on
        # (including the custom audio transforms)
        from examples import speech_to_text

    def build_test_ds_and_waveform(self, config):
        # fake must-speakers content
        mock_file_content = "TALK-ID\tTED-PRONOUN\n1\tHe\n2\tShe\n3\tHe\n"
        mock_file = mock_open(read_data=mock_file_content)
        # Patch the open function to return the mock file object
        with patch('builtins.open', mock_file):
            # setup the dataset class
            tgt_dict = Dictionary()
            tgt_lines = ["I like quokkas", "I like tortoises", "I like elephants"]
            for l in tgt_lines:
                tgt_dict.encode_line(l)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            synt_vowel_path = os.path.join(root_dir, 'waveform', "audio_data", "synt_vowel_i.wav")
            ds = SpeechToTextDataset(
                "train",
                True,
                config,
                [synt_vowel_path, synt_vowel_path, synt_vowel_path],
                [30, 100, 27],
                tgt_texts=tgt_lines,
                ids=["ted_1_1", "ted_2_1", "ted_1_2"],
                tgt_dict=tgt_dict)
            # initial_waveform
            source_waveform, _ = get_waveform(synt_vowel_path)
            return ds, source_waveform

    @patch('fairseq.data.audio.speech_to_text_dataset.fbank_features_from_waveform')
    def test_waveform_is_manipulated(self, mock_fbank_features_from_waveform):
        # avoid fbank computation
        mock_fbank_features_from_waveform.side_effect = lambda x, sample_rate, n_mel_bins: x
        ds, source_waveform = self.build_test_ds_and_waveform(MockS2TDataConfig(1.))
        # test correct alterations
        for i in range(3):
            self.assertFalse(np.allclose(ds[i][1], source_waveform))

    @patch('fairseq.data.audio.speech_to_text_dataset.fbank_features_from_waveform')
    def test_waveform_is_not_manipulated(self, mock_fbank_features_from_waveform):
        # avoid fbank computation
        mock_fbank_features_from_waveform.side_effect = lambda x, sample_rate, n_mel_bins: x
        ds, source_waveform = self.build_test_ds_and_waveform(MockS2TDataConfig(0.))
        # test correct alterations
        for i in range(3):
            self.assertTrue(np.allclose(ds[i][1], source_waveform))


if __name__ == '__main__':
    unittest.main()
