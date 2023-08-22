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

from fbk_uts.waveform.test_manipulation_base import measure_formants, TestSame


class TestSamePitch(TestSame):
    """
    Class to test PitchScalingSame for scaling M > M
    """
    def test_f0(self):
        pitch_scaling_random, _, new_sound = self.apply_manipulation()
        np.random.seed(2)
        expected_median_f0 = np.random.normal(140, 20)
        f0, f0_median = pitch_scaling_random._pitch_estimation(
            sound_object=new_sound, from_male=False)
        self.assertAlmostEqual(f0_median, expected_median_f0, delta=1)

    def test_formants(self):
        pitch_scaling_random, _, new_sound = self.apply_manipulation()
        actual_f1, actual_f2, actual_f3, actual_f4 = measure_formants(new_sound, 75, 250, 5000)
        self.assertAlmostEqual(actual_f1, self.original_f1, delta=50)
        self.assertAlmostEqual(actual_f2, self.original_f2, delta=50)
        self.assertAlmostEqual(actual_f3, self.original_f3, delta=50)
        self.assertAlmostEqual(actual_f4, self.original_f4, delta=50)


if __name__ == '__main__':
    unittest.main()
