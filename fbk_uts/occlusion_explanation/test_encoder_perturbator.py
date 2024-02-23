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

import numpy as np
import torch

from examples.speech_to_text.occlusion_explanation.perturbators.encoder_perturbator import \
    OcclusionFbankPerturbatorContinuous, OcclusionFbankPerturbatorDiscreteFrequency, \
    OcclusionFbankPerturbatorDiscreteTime, SlicOcclusionFbankPerturbator


class TestOcclusionFbankPerturbator(unittest.TestCase):
    def setUp(self):
        self.fbank = torch.ones(5, 10)   # Time, Channels
        self.original_fbank = self.fbank.clone()

    def test_from_config_dict_no_config_continuous(self):
        perturbator = OcclusionFbankPerturbatorContinuous.from_config_dict(config=None)
        self.assertEqual(perturbator.mask_probability, 0.5)
        self.assertEqual(perturbator.n_masks, 8000)

    def test_from_config_dict_continuous(self):
        cfg = {"fbank_occlusion": {"n_masks": 5, "p": 0.2}}
        perturbator = OcclusionFbankPerturbatorContinuous.from_config_dict(config=cfg)
        self.assertEqual(perturbator.mask_probability, 0.2)
        self.assertEqual(perturbator.n_masks, 5)

    def test_continuous_fbank(self):
        perturbator = OcclusionFbankPerturbatorContinuous(mask_probability=0.5, n_masks=5)
        mask, masked_fbank = perturbator(self.fbank)
        self.assertEqual(mask.shape, (5, 10))  # (time, channels)
        self.assertEqual(masked_fbank.shape, (5, 10))
        # Check if masking was effective
        self.assertFalse(torch.equal(self.fbank, masked_fbank))
        self.assertTrue(torch.any(masked_fbank == 0))
        self.assertTrue(torch.any(masked_fbank == 1))

    # Check that two different consecutive calls of perturbator generate two different masks
    def test_randomness(self):
        perturbator = OcclusionFbankPerturbatorContinuous(mask_probability=0.5, n_masks=5)
        mask1, masked_fbank1 = perturbator(self.fbank)
        mask2, masked_fbank2 = perturbator(self.fbank)
        self.assertFalse(torch.equal(mask1, mask2))
        self.assertFalse(torch.equal(masked_fbank1, masked_fbank2))

    # Evaluate the effectiveness of setting p = 0 or p = 1 in masking all or any of the values.
    def test_continuous_fbank_p_value(self):
        perturbator_0 = OcclusionFbankPerturbatorContinuous(mask_probability=0., n_masks=5)
        masks, masked_fbanks = perturbator_0(self.fbank)
        self.assertEqual(masked_fbanks.shape, (5, 10))
        self.assertFalse(torch.any(masked_fbanks == 0))
        self.assertTrue(torch.all(masked_fbanks == 1))

        perturbator_1 = OcclusionFbankPerturbatorContinuous(mask_probability=1., n_masks=5)
        masks, masked_fbanks = perturbator_1(self.fbank)
        self.assertEqual(masked_fbanks.shape, (5, 10))
        self.assertFalse(torch.any(masked_fbanks == 1))
        self.assertTrue(torch.all(masked_fbanks == 0))

    def test_discrete_fbank_time(self):
        torch.manual_seed(0)
        perturbator = OcclusionFbankPerturbatorDiscreteTime(mask_probability=0.5, n_masks=5)
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
        perturbator = OcclusionFbankPerturbatorDiscreteFrequency(mask_probability=0.5, n_masks=5)
        mask, masked_fbank = perturbator(self.fbank)
        expected_masked_fbank = torch.tensor(
            [[0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.]])
        self.assertEqual(mask.shape, (10,))
        self.assertTrue(torch.equal(masked_fbank, expected_masked_fbank))


class TestSlicFbankPerturbator(unittest.TestCase):
    def setUp(self):
        self.perturbator = SlicOcclusionFbankPerturbator(
            n_masks=5, mask_probability=0.5, segments_range=(2, 4), segments_step=1, slic_sigma=3)
        self.fbank = torch.tensor(
            [[100, 100, 100, 100, 100, 100],
             [1, 1, 1, 1, 1, 1],
             [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
             [30, 30, 30, 30, 30, 30]])

    def test_attributes(self):
        self.assertEqual(self.perturbator.min_segments, 2)
        self.assertEqual(self.perturbator.max_segments, 4)
        self.assertEqual(self.perturbator.segments_step, 1)
        self.assertEqual(self.perturbator.n_masks_per_segmentation, 1)
        self.assertEqual(self.perturbator.n_masks, 3)

    def test_slic_fbank_do_segmentation(self):
        segmentations = self.perturbator._do_segmentation(self.fbank)
        self.assertEqual(len(segmentations), 3)
        self.assertTrue(2 in segmentations.keys())
        self.assertTrue(3 in segmentations.keys())
        self.assertTrue(4 in segmentations.keys())

    def test_get_segments(self):
        test_index = 1
        perturb_index = 2
        # segmentation not available, computed for the first time
        segmentations = self.perturbator._do_segmentation(self.fbank)
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertEqual(type(segments), np.ndarray)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[4], segments))
        # Segmentation already computed and available
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[4], segments))

    # Test get_segments() with different perturb_index
    def test_get_segments_perturb_index(self):
        test_index = 1
        perturb_index = 5
        # segmentation not available, computed for the first time
        segmentations = self.perturbator._do_segmentation(self.fbank)
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertEqual(type(segments), np.ndarray)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[4], segments))
        # Segmentation already computed and available
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[4], segments))

    def test_call(self):
        test_index = 1
        perturb_index = 2
        torch.manual_seed(5)
        mask, masked_fbank = self.perturbator.__call__(self.fbank, test_index, perturb_index)
        self.assertEqual(mask.shape, self.fbank.shape)
        self.assertEqual(masked_fbank.shape, self.fbank.shape)
        self.assertTrue(torch.any(masked_fbank == 0))
        self.assertTrue(torch.any(masked_fbank == 1))


if __name__ == '__main__':
    unittest.main()
