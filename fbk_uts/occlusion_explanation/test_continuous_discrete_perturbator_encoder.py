# Copyright 2024 FBK

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
import torch

from examples.speech_to_text.occlusion_explanation.perturbators.discrete_fbank import (
    ContinuousOcclusionFbankPerturbator,
    DiscreteFrequencyOcclusionFbankPerturbator,
    DiscreteTimeOcclusionFbankPerturbator)


class TestOcclusionFbankPerturbator(unittest.TestCase):
    def setUp(self):
        self.fbank = torch.ones(5, 10)   # Time, Channels
        self.original_fbank = self.fbank.clone()

    def test_from_config_dict_no_config_continuous(self):
        perturbator = ContinuousOcclusionFbankPerturbator.from_config_dict(config=None)
        self.assertEqual(perturbator.mask_probability, 0.5)
        self.assertEqual(perturbator.n_masks, 8000)

    def test_from_config_dict_continuous(self):
        cfg = {"fbank_occlusion": {"n_masks": 5, "p": 0.2}}
        perturbator = ContinuousOcclusionFbankPerturbator.from_config_dict(config=cfg)
        self.assertEqual(perturbator.mask_probability, 0.2)
        self.assertEqual(perturbator.n_masks, 5)

    def test_continuous_fbank(self):
        perturbator = ContinuousOcclusionFbankPerturbator(mask_probability=0.5, n_masks=5)
        mask, masked_fbank = perturbator(self.fbank)
        self.assertEqual(mask.shape, (5, 10))  # (time, channels)
        self.assertEqual(masked_fbank.shape, (5, 10))
        # Check if masking was effective
        self.assertFalse(torch.equal(self.fbank, masked_fbank))
        self.assertTrue(torch.any(masked_fbank == 0))
        self.assertTrue(torch.any(masked_fbank == 1))

    # Check that two different consecutive calls of perturbator generate two different masks
    def test_randomness(self):
        perturbator = ContinuousOcclusionFbankPerturbator(mask_probability=0.5, n_masks=5)
        mask1, masked_fbank1 = perturbator(self.fbank)
        mask2, masked_fbank2 = perturbator(self.fbank)
        self.assertFalse(torch.equal(mask1, mask2))
        self.assertFalse(torch.equal(masked_fbank1, masked_fbank2))

    # Evaluate the effectiveness of setting p = 0 or p = 1 in masking all or any of the values.
    def test_p_value(self):

        def fake_random_with_zeros(shape):
            if isinstance(shape, torch.Size):
                shape = list(shape)
            else:
                shape = [shape]
            mask = torch.tensor(np.random.random(shape))
            if len(shape) == 2:
                mask[0][0] = 0
            elif len(shape) == 3:
                mask[0][0][0] = 0
            else:
                mask[0] = 0
            return mask

        with patch.object(torch, 'rand', new=fake_random_with_zeros):
            for perturbator_class in [
                    ContinuousOcclusionFbankPerturbator,
                    DiscreteTimeOcclusionFbankPerturbator,
                    DiscreteFrequencyOcclusionFbankPerturbator]:

                perturbator_0 = perturbator_class(mask_probability=0., n_masks=5)
                masks, masked_fbanks = perturbator_0(self.fbank)
                self.assertEqual(masked_fbanks.shape, (5, 10))
                self.assertFalse(torch.any(masked_fbanks == 0))
                self.assertTrue(torch.all(masked_fbanks == 1))

                perturbator_1 = perturbator_class(mask_probability=1., n_masks=5)
                masks, masked_fbanks = perturbator_1(self.fbank)
                self.assertEqual(masked_fbanks.shape, (5, 10))
                self.assertFalse(torch.any(masked_fbanks == 1))
                self.assertTrue(torch.all(masked_fbanks == 0))

    def test_discrete_fbank_time(self):
        torch.manual_seed(0)
        perturbator = DiscreteTimeOcclusionFbankPerturbator(mask_probability=0.5, n_masks=5)
        mask, masked_fbank = perturbator(self.fbank)
        expected_masked_fbank = torch.tensor(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        self.assertEqual(mask.shape, (5,))
        self.assertTrue(torch.equal(masked_fbank, expected_masked_fbank))

    def test_discrete_fbank_frequency(self):
        torch.manual_seed(0)
        perturbator = DiscreteFrequencyOcclusionFbankPerturbator(mask_probability=0.5, n_masks=5)
        mask, masked_fbank = perturbator(self.fbank)
        expected_masked_fbank = torch.tensor(
            [[0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.]])
        self.assertEqual(mask.shape, (10,))
        self.assertTrue(torch.equal(masked_fbank, expected_masked_fbank))


if __name__ == '__main__':
    unittest.main()
