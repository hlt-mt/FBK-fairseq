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

from fbk_uts.waveform.test_manipulation_base import measure_formants, TestOpposite


class TestOppositePitch(TestOpposite):
    """
    Class to test PitchScalingOpposite with 'pitch' as manipulation_type
    """
    def test_f0(self):
        pitch_scaling_opposite, _, new_sound = self.apply_manipulation("pitch", 44100, 1, "ted_1_123")
        np.random.seed(3)
        expected_median_f0 = np.random.normal(250, 17)
        f0, f0_median = pitch_scaling_opposite._pitch_estimation(
            sound_object=new_sound, from_male=False)
        self.assertAlmostEqual(f0_median, expected_median_f0, delta=1)

    def test_formants(self):
        pitch_scaling_opposite, _, new_sound = self.apply_manipulation("pitch", 44100, 1, "ted_1_123")
        actual_f1, actual_f2, actual_f3, actual_f4 = measure_formants(new_sound, 75, 250, 5000)
        # for f1, a higher difference is tollerable because the scaling of f0 perturbs the estimation of the f1.
        self.assertAlmostEqual(actual_f1, self.original_f1, delta=100)
        self.assertAlmostEqual(actual_f2, self.original_f2, delta=50)
        self.assertAlmostEqual(actual_f3, self.original_f3, delta=50)
        self.assertAlmostEqual(actual_f4, self.original_f4, delta=50)


class TestOppositeFormant(TestOpposite):
    """
    Class to test PitchScalingOpposite with 'formant' as manipulation_type
    """
    def test_f0(self):
        pitch_scaling_opposite, _, new_sound = self.apply_manipulation("formant", 44100, 1, "ted_1_123")
        np.random.seed(3)
        expected_median_f0 = np.random.normal(250, 17)
        f0, f0_median = pitch_scaling_opposite._pitch_estimation(
            sound_object=new_sound, from_male=False)
        self.assertAlmostEqual(f0_median, expected_median_f0, delta=1)

    def test_formants(self):
        pitch_scaling_opposite, _, new_sound = self.apply_manipulation("formant", 44100, 1, "ted_1_123")
        actual_f1, actual_f2, actual_f3, actual_f4 = measure_formants(new_sound, 100, 500, 5500)
        self.assertGreater(actual_f1, self.original_f1)
        self.assertGreater(actual_f2, self.original_f2)
        self.assertGreater(actual_f3, self.original_f3)
        self.assertGreater(actual_f4, self.original_f4)


if __name__ == '__main__':
    unittest.main()
