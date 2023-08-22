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
from typing import Tuple, Union
from unittest.mock import mock_open, patch

import numpy as np
import parselmouth
from parselmouth.praat import call

from examples.speech_to_text.data.waveform_transforms.pitch_manipulation import \
    PitchScalingOpposite, PitchScalingRandom, PitchScalingSame


def measure_formants(sound_object, pitch_floor, pitch_ceiling, maximum_formant) -> Tuple[
        Union[int, float, complex],
        Union[int, float, complex],
        Union[int, float, complex],
        Union[int, float, complex]]:
    point_process = call(sound_object, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    num_points = call(point_process, "Get number of points")
    formants = sound_object.to_formant_burg(
        time_step=0.0025,
        max_number_of_formants=5,
        maximum_formant=maximum_formant,
        window_length=0.025,
        pre_emphasis_from=50.0)
    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    for point in range(num_points):
        point += 1
        t = call(point_process, "Get time from index", point)
        f1 = formants.get_value_at_time(1, t)
        f2 = formants.get_value_at_time(2, t)
        f3 = formants.get_value_at_time(3, t)
        f4 = formants.get_value_at_time(4, t)
        if str(f1) != 'nan':
            f1_list.append(f1)
        if str(f2) != 'nan':
            f2_list.append(f2)
        if str(f3) != 'nan':
            f3_list.append(f3)
        if str(f4) != 'nan':
            f4_list.append(f4)
    f1_values = np.array(f1_list)
    f2_values = np.array(f2_list)
    f3_values = np.array(f3_list)
    f4_values = np.array(f4_list)
    original_f1_median = np.median(f1_values).item()
    original_f2_median = np.median(f2_values).item()
    original_f3_median = np.median(f3_values).item()
    original_f4_median = np.median(f4_values).item()
    return original_f1_median, original_f2_median, original_f3_median, original_f4_median


class TestBase(unittest.TestCase):
    """
    Base class for testing pitch manipulation classes
    """
    def setUp(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # synthetic wav file of a [i] vowel with pitch = 120 Hz (duration 1 s)
        file_path = os.path.join(root_dir, 'waveform', "audio_data", "synt_vowel_i.wav")
        self.sound_object = parselmouth.Sound(file_path)
        self.waveform = self.sound_object.values
        # Create a mock file object to simulate the gender TSV file
        mock_file_content = "TALK-ID\tTED-PRONOUN\n" "1\tHe\n" "2\tShe\n" "3\tHe\n" "4\tThey\n"
        self.mock_file = mock_open(read_data=mock_file_content)
        self.original_f1, self.original_f2, self.original_f3, self.original_f4 = measure_formants(
            self.sound_object, 75, 250, 5000)

    def apply_manipulation(self, **kwargs):
        pass


class TestOpposite(TestBase):
    """
    Class to test PitchScalingOpposite
    """
    def apply_manipulation(self, manipulation_type: str, sr: int, seed: int, talk_id: str):
        self.setUp()
        # Patch the open function to return the mock file object
        with patch('builtins.open', self.mock_file):
            pitch_scaling_opposite = PitchScalingOpposite(manipulation_type, 'must_speaker.txt', sr, 0.5, 0.5)
        random.seed(seed)  # seed 1 means that manipulation is performed, seed 2 means that it is not performed
        np.random.seed(3)  # with np.random.seed = 3, np.random.normal(250, 17) = 280.4066840483154
        new_waveform = pitch_scaling_opposite(self.waveform, talk_id)
        new_sound = parselmouth.Sound(new_waveform, sampling_frequency=sr, start_time=0.0)
        return pitch_scaling_opposite, new_waveform, new_sound


class TestRandom(TestBase):
    """
    Class to test PitchScalingRandom
    """
    def apply_manipulation(self, manipulation_type: str, seed1: int, seed2: int):
        self.setUp()
        # Patch the open function to return the mock file object
        with patch('builtins.open', self.mock_file):
            pitch_scaling_random = PitchScalingRandom(manipulation_type, 'must_speaker.txt', 44100, 1.0)
        random.seed(seed1)  # with random.seed(1), to_male = False, then M > F, with random.seed(4), to_male = True,
        # then M > M
        np.random.seed(seed2)  # with np.random.seed = 3, np.random.normal(250, 17) = 280.4066840483154
        new_waveform = pitch_scaling_random(self.waveform, "ted_1_123")
        new_sound = parselmouth.Sound(new_waveform, sampling_frequency=44100, start_time=0.0)
        return pitch_scaling_random, new_waveform, new_sound


class TestSame(TestBase):
    """
    Class to test PitchScalingSame
    """
    def apply_manipulation(self):
        self.setUp()
        with patch('builtins.open', self.mock_file):
            pitch_scaling_same = PitchScalingSame('must_speaker.txt', 44100, 0.7, 0.3)
        random.seed(1)  # with seed = 1, random.random = 0.13436424411240122, then manipulation is performed
        np.random.seed(2)  # with np.random.seed = 2, np.random.normal(140, 20) = 131.66484305189059
        new_waveform = pitch_scaling_same(self.waveform, "ted_1_123")
        new_sound = parselmouth.Sound(new_waveform, sampling_frequency=44100, start_time=0.0)
        return pitch_scaling_same, new_waveform, new_sound


if __name__ == '__main__':
    unittest.main()
