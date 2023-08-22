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

import numpy as np

from fbk_uts.waveform.test_manipulation_base import TestOpposite


class TestPitchScalingFunctions(TestOpposite):
    """
    Class to test single functions of the PitchScalingBase class.
    Opposite policy (formant manipulation) is used as manipualation class
    """
    def test_get_new_pitch(self):
        pitch_scaling_opposite, _, _ = self.apply_manipulation("formant", 44100, 1, "ted_1_123")
        new_pitch = pitch_scaling_opposite.get_new_pitch(False)
        self.assertGreaterEqual(new_pitch, 200)
        self.assertLessEqual(new_pitch, 300)

    def test_pitch_estimation(self):
        pitch_scaling_opposite, _, _ = self.apply_manipulation("formant", 44100, 1, "ted_1_123")
        f0, f0_median = pitch_scaling_opposite._pitch_estimation(
            sound_object=self.sound_object, from_male=True)
        expected_f0 = np.full(f0.selected_array['frequency'].shape, 120)
        self.assertTrue(np.allclose(f0.selected_array['frequency'], expected_f0, atol=1))
        self.assertAlmostEqual(f0_median, 120, delta=1)

    def test_they_gender(self):
        _, new_waveform, _ = self.apply_manipulation("formant", 44100, 1, "ted_4_1")
        self.assertTrue(np.allclose(self.waveform, new_waveform))

    def test_not_applicable_probability(self):
        pitch_scaling_opposite, new_waveform, _ = self.apply_manipulation("formant", 44100, 2, "ted_1_123")
        self.assertTrue(pitch_scaling_opposite._not_applicable("He", "ted_3_45"))
        self.assertTrue(np.allclose(self.waveform, new_waveform))


if __name__ == '__main__':
    unittest.main()
