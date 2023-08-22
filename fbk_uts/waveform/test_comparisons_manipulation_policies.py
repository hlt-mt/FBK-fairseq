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

from fbk_uts.waveform.test_manipulation_base import measure_formants, TestRandom, TestOpposite, TestSame


class TestSameEqualToRandom(unittest.TestCase):
    """
    Class to test that Same gives the same outcomes as Random (pitch, M > M)
    """
    @staticmethod
    def get_results():
        pitch_scaling_same, _, new_sound_same = TestSame().apply_manipulation()
        pitch_scaling_random, _, new_sound_random = TestRandom().apply_manipulation('pitch', 4, 2)
        return pitch_scaling_same, new_sound_same, pitch_scaling_random, new_sound_random

    def test_f0(self):
        pitch_scaling_same, new_sound_same, pitch_scaling_random, new_sound_random = self.get_results()
        _, f0_median1 = pitch_scaling_same._pitch_estimation(
            sound_object=new_sound_same, from_male=True)
        _, f0_median2 = pitch_scaling_random._pitch_estimation(
            sound_object=new_sound_random, from_male=True)
        self.assertAlmostEqual(f0_median1, f0_median2, delta=0.005)

    def test_formants(self):
        _, new_sound_same, _, new_sound_random = self.get_results()
        f1_1, f2_1, f3_1, f4_1 = measure_formants(new_sound_same, 75, 250, 5000)
        f1_2, f2_2, f3_2, f4_2 = measure_formants(new_sound_random, 75, 250, 5000)
        self.assertAlmostEqual(f1_1, f1_2, delta=0.01)
        self.assertAlmostEqual(f2_1, f2_2, delta=0.01)
        self.assertAlmostEqual(f3_1, f3_2, delta=0.01)
        self.assertAlmostEqual(f4_1, f4_2, delta=0.01)


class TestOppositeEqualToRandomFormant(unittest.TestCase):
    """
    Class to test that Opposite (formant) gives the same outcomes as Random (formant, M > F)
    """
    @staticmethod
    def get_results():
        pitch_scaling_opposite, _, new_sound_opposite = TestOpposite().apply_manipulation(
            "formant", 44100, 1, "ted_1_123")
        pitch_scaling_random, _, new_sound_random = TestRandom().apply_manipulation('formant', 1, 3)
        return pitch_scaling_opposite, new_sound_opposite, pitch_scaling_random, new_sound_random

    def test_f0(self):
        pitch_scaling_opposite, new_sound_opposite, pitch_scaling_random, new_sound_random = self.get_results()
        _, f0_median1 = pitch_scaling_opposite._pitch_estimation(
            sound_object=new_sound_opposite, from_male=False)
        _, f0_median2 = pitch_scaling_random._pitch_estimation(
            sound_object=new_sound_random, from_male=False)
        self.assertAlmostEqual(f0_median1, f0_median2, delta=0.01)

    def test_formants(self):
        _, new_sound_opposite, _, new_sound_random = self.get_results()
        f1_1, f2_1, f3_1, f4_1 = measure_formants(new_sound_opposite, 100, 500, 5500)
        f1_2, f2_2, f3_2, f4_2 = measure_formants(new_sound_random, 100, 500, 5500)
        self.assertAlmostEqual(f1_1, f1_2, delta=0.01)
        self.assertAlmostEqual(f2_1, f2_2, delta=0.01)
        self.assertAlmostEqual(f3_1, f3_2, delta=0.01)
        self.assertAlmostEqual(f4_1, f4_2, delta=0.01)


class TestOppositeEqualToRandomPitch(unittest.TestCase):
    """
    Class to test that Opposite (pitch) gives the same outcomes as Random (pitch, M > F)
    """
    @staticmethod
    def get_results():
        pitch_scaling_opposite, _, new_sound_opposite = TestOpposite().apply_manipulation(
            "pitch", 44100, 1, "ted_1_123")
        pitch_scaling_random, _, new_sound_random = TestRandom().apply_manipulation('pitch', 1, 3)
        return pitch_scaling_opposite, new_sound_opposite, pitch_scaling_random, new_sound_random

    def test_f0(self):
        pitch_scaling_opposite, new_sound_opposite, pitch_scaling_random, new_sound_random = self.get_results()
        _, f0_median1 = pitch_scaling_opposite._pitch_estimation(
            sound_object=new_sound_opposite, from_male=False)
        _, f0_median2 = pitch_scaling_random._pitch_estimation(
            sound_object=new_sound_random, from_male=False)
        self.assertAlmostEqual(f0_median1, f0_median2, delta=0.01)

    def test_formants(self):
        _, new_sound_opposite, _, new_sound_random = self.get_results()
        f1_1, f2_1, f3_1, f4_1 = measure_formants(new_sound_opposite, 75, 250, 5000)
        f1_2, f2_2, f3_2, f4_2 = measure_formants(new_sound_random, 75, 250, 5000)
        self.assertAlmostEqual(f1_1, f1_2, delta=0.01)
        self.assertAlmostEqual(f2_1, f2_2, delta=0.01)
        self.assertAlmostEqual(f3_1, f3_2, delta=0.01)
        self.assertAlmostEqual(f4_1, f4_2, delta=0.01)


class TestSameEqualToOppositePitch(unittest.TestCase):
    """
    Class to test that Same (then also Random pitch M > M) gives the same outcomes
    as Opposite pitch (then also Random pitch M > F) wrt formants
    """
    @staticmethod
    def get_results():
        pitch_scaling_same, _, new_sound_same = TestSame().apply_manipulation()
        pitch_scaling_opposite, _, new_sound_opposite = TestOpposite().apply_manipulation(
            "pitch", 44100, 1, "ted_1_123")
        return pitch_scaling_same, new_sound_same, pitch_scaling_opposite, new_sound_opposite

    def test_formants(self):
        _, new_sound_same, _, new_sound_opposite = self.get_results()
        f1_1, f2_1, f3_1, f4_1 = measure_formants(new_sound_same, 75, 250, 5000)
        f1_2, f2_2, f3_2, f4_2 = measure_formants(new_sound_opposite, 75, 250, 5000)
        # for f1, a higher delta is tollerable because in Opposite the scaling of f0 perturbs the estimation of the f1.
        self.assertAlmostEqual(f1_1, f1_2, delta=70)
        self.assertAlmostEqual(f2_1, f2_2, delta=25)
        self.assertAlmostEqual(f3_1, f3_2, delta=25)
        self.assertAlmostEqual(f4_1, f4_2, delta=25)


class TestSameDifferentThanOppositeFormant(unittest.TestCase):
    """
    Class to test that Same gives (then also Random pitch M > M) gives different outcomes
    than Opposite formant (then also Random formant M > F)
    """
    @staticmethod
    def get_results():
        pitch_scaling_same, _, new_sound_same = TestSame().apply_manipulation()
        pitch_scaling_opposite, _, new_sound_opposite = TestOpposite().apply_manipulation(
            "formant", 44100, 1, "ted_1_123")
        return pitch_scaling_same, new_sound_same, pitch_scaling_opposite, new_sound_opposite

    def test_f0(self):
        pitch_scaling_same, new_sound_same, pitch_scaling_opposite, new_sound_opposite = self.get_results()
        _, f0_median1 = pitch_scaling_same._pitch_estimation(
            sound_object=new_sound_same, from_male=True)
        _, f0_median2 = pitch_scaling_opposite._pitch_estimation(
            sound_object=new_sound_opposite, from_male=False)
        self.assertNotAlmostEqual(f0_median1, f0_median2, delta=140)

    def test_formants(self):
        _, new_sound_same, _, new_sound_opposite = self.get_results()
        f1_1, f2_1, f3_1, f4_1 = measure_formants(new_sound_same, 75, 250, 5000)
        f1_2, f2_2, f3_2, f4_2 = measure_formants(new_sound_opposite, 100, 500, 5500)
        self.assertNotAlmostEqual(f1_1, f1_2, delta=120)
        self.assertNotAlmostEqual(f2_1, f2_2, delta=200)
        self.assertNotAlmostEqual(f3_1, f3_2, delta=200)
        self.assertNotAlmostEqual(f4_1, f4_2, delta=200)


if __name__ == '__main__':
    unittest.main()
