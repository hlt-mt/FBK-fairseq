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
from parselmouth.praat import call

from examples.speech_to_text.data.waveform_transforms.pitch_manipulation import \
    PitchScalingOpposite, \
    PitchScalingRandom


class TestPitchFormantManipulation(unittest.TestCase):
    def setUp(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # synthetic wav file of a [i] vowel with pitch = 120 Hz (duration 1 s)
        file_path = os.path.join(root_dir, 'waveform', "audio_data", "synt_vowel_i.wav")
        self.sound_object = parselmouth.Sound(file_path)
        self.waveform = self.sound_object.values
        # Create a mock file object to simulate the gender TSV file
        mock_file_content = "TALK-ID\tTED-PRONOUN\n" "1\tHe\n" "2\tShe\n" "3\tHe\n" "4\tThey\n"
        mock_file = mock_open(read_data=mock_file_content)
        # Patch the open function to return the mock file object
        with patch('builtins.open', mock_file):
            self.pitch_scaling_opposite = PitchScalingOpposite('must_speaker.txt', 44100, 0.3, 0.7)
            self.pitch_scaling_random = PitchScalingRandom('must_speaker.txt', 16000, 1.0)

    def test_get_new_pitch(self):
        new_pitch = self.pitch_scaling_opposite.get_new_pitch(False)
        self.assertGreaterEqual(new_pitch, 200)
        self.assertLessEqual(new_pitch, 300)

    def test_pitch_estimation(self):
        f0, f0_median = self.pitch_scaling_opposite._pitch_estimation(
            sound_object=self.sound_object,
            from_male=True)
        expected_f0 = np.full(f0.selected_array['frequency'].shape, 120)
        self.assertTrue(np.allclose(f0.selected_array['frequency'], expected_f0, rtol=1))
        self.assertAlmostEqual(f0_median, 120, delta=1)

    # Verifying the __call__ of the manipulation class
    def test_manipulation(self):
        # getting formant values of the original audio file
        pointProcess = call(self.sound_object, "To PointProcess (periodic, cc)", 75, 250)
        numPoints = call(pointProcess, "Get number of points")

        formants = self.sound_object.to_formant_burg(
            time_step=0.0025,
            max_number_of_formants=5,
            maximum_formant=5000.0,
            window_length=0.025,
            pre_emphasis_from=50.0)

        f1_list = []
        f2_list = []
        f3_list = []
        f4_list = []
        for point in range(numPoints):
            point += 1
            t = call(pointProcess, "Get time from index", point)
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

        # Getting formant values of the manipulated audio file
        random.seed(1)  # with seed = 1, random.random = 0.13436424411240122
        np.random.seed(3)  # with np.random.seed = 3, np.random.normal(250, 17) = 280.4066840483154
        new_waveform = self.pitch_scaling_opposite(self.waveform, "ted_1_123")
        new_sound = parselmouth.Sound(new_waveform, sampling_frequency=44100, start_time=0.0)
        pointProcess = call(new_sound, "To PointProcess (periodic, cc)", 100, 500)
        numPoints = call(pointProcess, "Get number of points")

        formants = new_sound.to_formant_burg(
            time_step=0.0025,
            max_number_of_formants=5,
            maximum_formant=5500.0,
            window_length=0.025,
            pre_emphasis_from=50.0)

        f1_list = []
        f2_list = []
        f3_list = []
        f4_list = []
        for point in range(numPoints):
            point += 1
            t = call(pointProcess, "Get time from index", point)
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

        actual_f1_median = np.median(f1_values).item()
        actual_f2_median = np.median(f2_values).item()
        actual_f3_median = np.median(f3_values).item()
        actual_f4_median = np.median(f4_values).item()

        self.assertGreater(actual_f1_median, original_f1_median)
        self.assertGreater(actual_f2_median, original_f2_median)
        self.assertGreater(actual_f3_median, original_f3_median)
        self.assertGreater(actual_f4_median, original_f4_median)

        # Checking F0
        np.random.seed(3)
        expected_median_f0 = np.random.normal(250, 17)
        f0, f0_median = self.pitch_scaling_opposite._pitch_estimation(
            sound_object=new_sound,
            from_male=False)  # the new manipulated sound is supposed to be of a female speaker
        self.assertAlmostEqual(f0_median, expected_median_f0, delta=1)

    def test_they_gender(self):
        random.seed(2)
        new_waveform = self.pitch_scaling_opposite(self.waveform, "ted_4_1")
        self.assertTrue(np.allclose(self.waveform, new_waveform))

    def test_non_applicable(self):
        random.seed(2)  # with seed = 2, random.random() = 0.9560342718892494
        self.assertTrue(self.pitch_scaling_opposite._not_applicable("He", "ted_3_45"))

    def test_probability(self):
        random.seed(2)
        new_waveform = self.pitch_scaling_opposite(self.waveform, "ted_3_45")
        self.assertTrue(np.allclose(self.waveform, new_waveform))

    # Verify that for audio segments which do not contain speech portions, the transformation is not applied
    def test_non_voices(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # wav file extracted from MuST-C and containing applauses
        file_path = os.path.join(root_dir, 'waveform', "audio_data", "applause_ted.wav")
        original_sound_object = parselmouth.Sound(file_path).values
        new_sound_object = self.pitch_scaling_random(original_sound_object, "ted_4_1")
        self.assertTrue(np.allclose(original_sound_object, new_sound_object))
        

if __name__ == '__main__':
    unittest.main()
