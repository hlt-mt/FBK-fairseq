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
from typing import Set, Optional, Dict
from unittest.mock import patch

import numpy as np
import torch

from fairseq.data import Dictionary
from fairseq.data.audio.feature_transforms import register_audio_feature_transform, AudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig, SpeechToTextDataset


@register_audio_feature_transform("fake")
class FakeWaveformTransform(AudioFeatureTransform):
    @property
    def extra_args(self) -> Set[str]:
        return {"ids"}

    def __call__(self, wav, ids=None):
        if ids == "2":
            return wav * 2
        else:
            return wav

    @classmethod
    def from_config_dict(cls, config: Optional[Dict] = None):
        return cls()


class MockS2TDataConfig(S2TDataConfig):
    def __init__(self):
        self.config = {}

    def is_input_waveform(self):
        return True

    def waveform_sample_rate(self):
        return 16000

    def get_waveform_transforms(self, split, is_train):
        return {"transforms": ["fake"]}


class SpeechToTextDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.tgt_dict = Dictionary()
        tgt_lines = ["I like quokkas", "I like tortoises", "I like elephants"]
        for l in tgt_lines:
            self.tgt_dict.encode_line(l)
        self.ds = SpeechToTextDataset(
            "train",
            True,
            MockS2TDataConfig(),
            ["f1.wav", "f2.wav", "f3.wav"],
            [30, 100, 27],
            tgt_texts=tgt_lines,
            ids=["1", "2", "3"],
            tgt_dict=self.tgt_dict)

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    @patch('fairseq.data.audio.speech_to_text_dataset.fbank_features_from_waveform')
    def test_waveform_transform(self, mock_fbank_features_from_waveform, mock_get_features_or_waveform):
        mock_fbank_features_from_waveform.side_effect = lambda x, sample_rate, n_mel_bins: x
        mock_get_features_or_waveform.return_value = np.array([[1., 1., 1., 1.]])
        self.assertListEqual(self.ds[0][1].tolist(), [1., 1., 1., 1.])
        self.assertListEqual(self.ds[1][1].tolist(), [2., 2., 2., 2.])
        self.assertListEqual(self.ds[2][1].tolist(), [1., 1., 1., 1.])


if __name__ == '__main__':
    unittest.main()
