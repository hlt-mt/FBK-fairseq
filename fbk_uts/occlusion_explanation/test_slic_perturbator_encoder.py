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

import copy
import unittest

import numpy as np
import torch

from examples.speech_to_text.occlusion_explanation.perturbators.slic_fbank import \
    SlicOcclusionFbankPerturbatorDynamicSegments, SlicOcclusionFbankPerturbatorFixedSegments


class TestFixedSlicPerturbator(unittest.TestCase):
    def setUp(self):
        self.perturbator = SlicOcclusionFbankPerturbatorFixedSegments(
            n_masks=23,
            mask_probability=0.2,
            n_segments=[1, 4],
            slic_sigma=1,
            compactness=0.1)
        self.fbank = torch.tensor(
            [[-200, -222, -230, 100, 100, 100, 50, 43, 21, 23, 1],
             [-240, 1, 1, 1, 1, 1, 32, 54, 23, 54, 0.5],
             [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 10, 11, 67, 10, 45],
             [30, 30, 30, 30, 30, 30, 243, 12, 43, 23, 34]])

    def test_attributes(self):
        self.assertEqual(self.perturbator.n_segments, [1, 4])
        self.assertEqual(self.perturbator.n_segmentations, 2)
        self.assertEqual(self.perturbator.n_masks_per_segmentation, 11)
        self.assertEqual(self.perturbator.n_masks, 22)
        with self.assertRaises(AttributeError):
            _ = self.perturbator.reference_duration
        with self.assertRaises(AttributeError):
            _ = self.perturbator.threshold_duration

    # Test get_granularity_level() with n_masks as int
    def test_get_granularity_level_int(self):
        perturb_indices = [34, 12, 23, 21]
        expected = [1, 1, 0, 1]
        for i, e in zip(perturb_indices, expected):
            granularity_level = self.perturbator.get_granularity_level(i)
            self.assertEqual(granularity_level, e)

    # Test get_granularity_level() with n_masks as list
    def test_get_granularity_level_list(self):
        perturbator = SlicOcclusionFbankPerturbatorFixedSegments(
            n_masks=[2, 8, 12],
            mask_probability=0.2,
            n_segments=[1, 4, 8],
            slic_sigma=1,
            compactness=0.1)
        perturb_indices = [34, 1, 4, 21]
        expected = [2, 0, 1, 2]
        for i, e in zip(perturb_indices, expected):
            granularity_level = perturbator.get_granularity_level(i)
            self.assertEqual(granularity_level, e)

    def test_slic_fbank_do_segmentation(self):
        segment_dict = self.perturbator._do_segmentation(self.fbank)
        self.assertEqual(len(segment_dict), 2)
        self.assertTrue(0 in segment_dict.keys())
        self.assertTrue(1 in segment_dict.keys())
        self.assertFalse(2 in segment_dict.keys())
        self.assertEqual(np.max(segment_dict[0]), 1)
        self.assertEqual(np.max(segment_dict[1]), 4)

    def test_get_segments(self):
        test_index = 1
        perturb_index = 12
        # segmentation not available, computed for the first time
        segmentations = self.perturbator._do_segmentation(self.fbank)
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertEqual(type(segments), np.ndarray)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[1], segments))
        # Segmentation already computed and available
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[1], segments))

    # Test get_segments() with different perturb_index
    def test_get_segments_perturb_index(self):
        test_index = 1
        perturb_index = 6
        # segmentation not available, computed for the first time
        segmentations = self.perturbator._do_segmentation(self.fbank)
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertEqual(type(segments), np.ndarray)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[0], segments))
        # Segmentation already computed and available
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[0], segments))

    # Test call() with n_masks as int
    def test_call_int(self):
        test_index = 1
        perturb_index = 12
        torch.manual_seed(5)
        mask, masked_fbank = self.perturbator.__call__(self.fbank, test_index, perturb_index)
        self.assertEqual(mask.shape, self.fbank.shape)
        self.assertEqual(masked_fbank.shape, self.fbank.shape)
        self.assertTrue(torch.any(masked_fbank == 0))
        self.assertTrue(torch.any(masked_fbank == 1))

    # Test call() with n_masks as list
    def test_call_list(self):
        perturbator = SlicOcclusionFbankPerturbatorFixedSegments(
            n_masks=[2, 8, 12],
            mask_probability=0.2,
            n_segments=[1, 4, 8],
            slic_sigma=1,
            compactness=0.1)
        test_index = 1
        perturb_index = 25
        torch.manual_seed(5)
        mask, masked_fbank = perturbator.__call__(self.fbank, test_index, perturb_index)
        self.assertEqual(mask.shape, self.fbank.shape)
        self.assertEqual(masked_fbank.shape, self.fbank.shape)
        self.assertTrue(torch.any(masked_fbank == 0))
        self.assertTrue(torch.any(masked_fbank == 1))

    def test_get_n_segments(self):
        n_segments = self.perturbator.get_n_segments(n_frames=12, n_segments=20)
        self.assertEqual(n_segments, 20)

    # check that at least 1 segment is guaranteed
    def test_get_n_segments_zero(self):
        n_segments = self.perturbator.get_n_segments(n_frames=4, n_segments=2)
        self.assertEqual(n_segments, 2)

    def test_parse_custom_args(self):
        fbank_occlusion_config = {
            "n_masks": 20, "reference_duration": 40, "threshold_duration": 750}
        add_config = self.perturbator._parse_custom_args(fbank_occlusion_config)
        self.assertEqual(add_config, {})


class TestDynamicSlicPerturbator(unittest.TestCase):
    def setUp(self):
        self.perturbator = SlicOcclusionFbankPerturbatorDynamicSegments(
            n_masks=32,
            mask_probability=0.5,
            n_segments=[1, 4, 8],
            slic_sigma=1,
            compactness=0.1,
            reference_duration=10)
        self.fbank = torch.tensor(
            [[0, 2, 45, 100, 100, 100, 50, 43, 21, 23, 1],
             [1, 23, 23, 1, 1, 1, 32, 54, 23, 54, 0.5],
             [22, 1.5, 22, 0, 1.5, 1.5, 10, 11, 67, 10, 45],
             [44, 32, 45, 30, 65, 90, 243, 12, 43, 23, 34]])

    def test_attributes(self):
        self.assertEqual(self.perturbator.n_segments, [1, 4, 8])
        self.assertEqual(self.perturbator.reference_duration, 10)
        self.assertEqual(self.perturbator.threshold_duration, None)
        self.assertEqual(self.perturbator.n_segmentations, 3)
        self.assertEqual(self.perturbator.n_masks_per_segmentation, 10)
        self.assertEqual(self.perturbator.n_masks, 30)

    def test_slic_fbank_do_segmentation(self):
        segment_dict = self.perturbator._do_segmentation(self.fbank)
        self.assertEqual(len(segment_dict), 3)
        self.assertTrue(0 in segment_dict.keys())
        self.assertTrue(1 in segment_dict.keys())
        self.assertTrue(2 in segment_dict.keys())
        self.assertEqual(np.max(segment_dict[0]), 1)  # 1 segment instead of 0
        self.assertEqual(np.max(segment_dict[1]), 2)  # 2 segments
        self.assertEqual(np.max(segment_dict[2]), 3)  # 3 segments

    def test_get_segments(self):
        test_index = 1
        perturb_index = 32
        # segmentation not available, computed for the first time
        segmentations = self.perturbator._do_segmentation(self.fbank)
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertEqual(type(segments), np.ndarray)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[0], segments))
        # Segmentation already computed and available
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[0], segments))

    # Test get_segments() with different perturb_index
    def test_get_segments_perturb_index(self):
        test_index = 1
        perturb_index = 43
        # segmentation not available, computed for the first time
        segmentations = self.perturbator._do_segmentation(self.fbank)
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertEqual(type(segments), np.ndarray)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[1], segments))
        # Segmentation already computed and available
        segments = self.perturbator.get_segments(self.fbank, test_index, perturb_index)
        self.assertTrue(segmentations, self.perturbator.test_index_to_segment[test_index])
        self.assertTrue(np.array_equal(segmentations[1], segments))

    def test_call(self):
        test_index = 1
        perturb_index = 52
        torch.manual_seed(5)
        mask, masked_fbank = self.perturbator.__call__(self.fbank, test_index, perturb_index)
        self.assertEqual(mask.shape, self.fbank.shape)
        self.assertEqual(masked_fbank.shape, self.fbank.shape)
        self.assertTrue(torch.any(masked_fbank == 0))
        self.assertTrue(torch.any(masked_fbank == 1))

    def test_get_n_segments_below_threshold(self):
        # with threshold_duration not set (default behaviour)
        n_segments = self.perturbator.get_n_segments(n_frames=12, n_segments=20)
        self.assertEqual(n_segments, 24)
        # with threshold_duration set (does not change behaviour if we are below the threshold)
        perturbator_with_threshold = copy.deepcopy(self.perturbator)
        perturbator_with_threshold.threshold_duration = 20
        n_segments = perturbator_with_threshold.get_n_segments(n_frames=12, n_segments=20)
        self.assertEqual(n_segments, 24)

    def test_get_n_segments_above_threshold(self):
        # with threshold_duration not set (default behaviour)
        n_segments = self.perturbator.get_n_segments(n_frames=24, n_segments=20)
        self.assertEqual(n_segments, 48)
        # with threshold_duration set
        perturbator_with_threshold = copy.deepcopy(self.perturbator)
        perturbator_with_threshold.threshold_duration = 20
        n_segments = perturbator_with_threshold.get_n_segments(n_frames=24, n_segments=20)
        self.assertEqual(n_segments, 40)

    # check that at least 1 segment is guaranteed
    def test_get_n_segments_zero(self):
        n_segments = self.perturbator.get_n_segments(n_frames=4, n_segments=2)
        self.assertEqual(n_segments, 1)

    def test_parse_custom_args(self):
        fbank_occlusion_config = {
            "n_masks": 20, "reference_duration": 40, "threshold_duration": 750}
        add_config = self.perturbator._parse_custom_args(fbank_occlusion_config)
        self.assertEqual(add_config, {"reference_duration": 40, "threshold_duration": 750})

    def test_parse_custom_args_empty(self):
        fbank_occlusion_config = {"n_masks": 20}
        add_config = self.perturbator._parse_custom_args(fbank_occlusion_config)
        self.assertEqual(add_config, {"reference_duration": 500, "threshold_duration": None})


if __name__ == '__main__':
    unittest.main()
