# Copyright 2022 FBK
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

from examples.speech_to_text.data.speech_to_text_dataset_multimodal import SpeechToTextDatasetMultimodal
from fbk_uts.base_utilities import BaseSpeechTestCase


class DualEncoderDatasetSetup(BaseSpeechTestCase):
    def setUp(self):
        self.init_sample_dataset(SpeechToTextDatasetMultimodal)


class DualEncoderDatasetTestCase(DualEncoderDatasetSetup, unittest.TestCase):

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_collater(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        self.assertTrue("context_tokens" in samples["net_input"])
        self.assertTrue("context_lengths" in samples["net_input"])
        # To date, only support transcript as context
        self.assertEqual(samples["transcript"].tolist(), samples["net_input"]["context_tokens"].tolist())
        self.assertEqual(samples["transcript_lengths"].tolist(), samples["net_input"]["context_lengths"].tolist())

    def test_empty_samples(self):
        samples = self.ds.collater([])
        self.assertEqual(samples, {})


if __name__ == '__main__':
    unittest.main()
