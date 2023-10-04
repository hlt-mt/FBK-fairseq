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
import numpy as np
import unittest

import parselmouth

from examples.speech_to_text.data.waveform_transforms.vtlp import VTLP


class TestVTLP(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # synthetic wav file of a [i] vowel with pitch = 120 Hz (duration 1 s)
        synt_wav = os.path.join(self.root_dir, 'waveform', "audio_data", "synt_vowel_i.wav")
        self.waveform = parselmouth.Sound(synt_wav).values
        # perturbed waveform obtained with seed = 3 using speechaugs.VTLP(),
        # the previous code that served as our inspiration
        reference_augmented_file = os.path.join(
            self.root_dir, 'waveform', "audio_data", "perturbed_synt_i.npy")
        self.reference_augmented_waveform = np.load(reference_augmented_file)

    def test_compare_with_reference(self):
        # Checks that the current implementation returns the same result of the other available
        # implementations
        vtlp_transform = VTLP(
            sr=44100, warp_factor_boundaries=(0.9, 1.1), sampling_type="uniform", boundary_freq=4800)
        np.random.seed(3)
        augmented_waveform = vtlp_transform(self.waveform)
        # atol set to avoid floating-point precision issues in the comparison
        self.assertTrue(
            np.allclose(augmented_waveform, self.reference_augmented_waveform, atol=1e-04))

    def test_compare_sampling_type(self):
        # Checks that the different sampling types provide different results
        vtlp_transform_random = VTLP(
            sr=44100, warp_factor_boundaries=(0.9, 1.1), sampling_type="uniform", boundary_freq=4800)
        np.random.seed(3)
        random_augmented_waveform = vtlp_transform_random(self.waveform)
        vtlp_transform_uniform = VTLP(
            sr=44100, warp_factor_boundaries=(0.9, 1.1), sampling_type="random", boundary_freq=4800)
        np.random.seed(3)
        uniform_augmented_waveform = vtlp_transform_uniform(self.waveform)
        self.assertFalse(np.allclose(uniform_augmented_waveform, random_augmented_waveform))

    def test_get_scale_factor(self):
        # Ensures that if the scale factor falls outside the range defined by warp_factor_boundaries,
        # it is reset to the nearest boundary value.
        np.random.seed(100)  # np.random.normal(1, (1.1 - 0.9) / 2) = 0.8250234526945301
        scale_factor = VTLP().get_scale_factor("random", 0.9, 1.1)
        self.assertEqual(scale_factor, 0.9)
