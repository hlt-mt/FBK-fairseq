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
import random
from unittest.mock import mock_open, patch

import numpy as np
import parselmouth

from examples.speech_to_text.data.waveform_transforms.pitch_manipulation import PitchScalingOpposite


class TestNonVoicedSegments(unittest.TestCase):
    """
    Verify that for audio segments which do not contain speech portions, the transformation is not applied
    """
    def test_non_voiced(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # wav file extracted from MuST-C and containing applauses
        file_path = os.path.join(root_dir, 'waveform', "audio_data", "applause_ted.wav")
        original_sound_object = parselmouth.Sound(file_path)
        self.original_waveform = original_sound_object.values
        # Create a mock file object to simulate the gender TSV file
        mock_file_content = "TALK-ID\tTED-PRONOUN\n" "1\tHe\n" "2\tShe\n" "3\tHe\n" "4\tThey\n"
        mock_file = mock_open(read_data=mock_file_content)
        with patch('builtins.open', mock_file):
            self.pitch_scaling_opposite = PitchScalingOpposite('formant', 'must_speaker.txt', 16000, 0.5, 0.5)
        random.seed(1)  # seed 1 means that manipulation is performed, seed 2 means that it is not performed
        new_waveform = self.pitch_scaling_opposite(self.original_waveform, "ted_1_123")
        self.assertTrue(np.allclose(self.original_waveform, new_waveform))


if __name__ == '__main__':
    unittest.main()
