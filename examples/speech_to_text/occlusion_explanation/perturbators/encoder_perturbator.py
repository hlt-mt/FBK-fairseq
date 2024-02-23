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

import logging
from typing import Dict, Tuple
import numpy as np

from skimage.segmentation import slic

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.perturbators import \
    register_perturbator, OcclusionFbankPerturbator


LOGGER = logging.getLogger(__name__)


@register_perturbator("continuous_fbank")
class OcclusionFbankPerturbatorContinuous(OcclusionFbankPerturbator):
    """
    Class for implementing continuous occlusion perturbations.
    Under this method, each value in the input data is perturbed independently of the others.
    """
    def __init__(self, mask_probability: float, n_masks: int):
        self.mask_probability = mask_probability
        LOGGER.info(f"Masking fbanks with probability of {self.mask_probability}.")
        self.n_masks = n_masks
        LOGGER.info(f"{self.n_masks} masks are used.")

    @classmethod
    def from_config_dict(cls, config: Dict = None):
        if config is None:
            return cls(mask_probability=0.5, n_masks=8000)
        else:
            _config = config
            fbank_occlusion = _config.get("fbank_occlusion", {})
            mask_probability = fbank_occlusion.get("p", 0.5)
            assert 0. < mask_probability <= 1.
            n_masks = fbank_occlusion.get("n_masks", 8000)
            return cls(mask_probability=mask_probability, n_masks=n_masks)

    def __call__(self, fbank: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Builds mask and performs masking for filterbank fed into the encoder.
        Args:
            - fbank: Tensor: a single filterbank of shape (time, channels)
        Returns:
            - mask: Tensor of shape (time, channels)
            - masked_fbank: masked version of the original fbank with shape (time, channels)
        """
        mask = (torch.rand(fbank.size()) > self.mask_probability).to(fbank.dtype)
        masked_fbank = fbank * mask
        return mask, masked_fbank


@register_perturbator("discrete_fbank_time")
class OcclusionFbankPerturbatorDiscreteTime(OcclusionFbankPerturbatorContinuous):
    """
    Class for implementing occlusion perturbations discrete in the time domain.
    In this method, entire time frames in the input data are perturbed.
    """
    def __call__(self, fbank: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        mask = (torch.rand(fbank.size(0)) > self.mask_probability).to(fbank.dtype)
        masked_fbank = fbank * mask.unsqueeze(1)
        return mask, masked_fbank


@register_perturbator("discrete_fbank_frequency")
class OcclusionFbankPerturbatorDiscreteFrequency(OcclusionFbankPerturbatorContinuous):
    """
    Class for implementing occlusion perturbations discrete in the frequency domain.
    In this method, entire frequency rows in the input data are perturbed.
    """
    def __call__(self, fbank: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        mask = (torch.rand(fbank.size(1)) > self.mask_probability).to(fbank.dtype)
        masked_fbank = fbank * mask.unsqueeze(0)
        return mask, masked_fbank


@register_perturbator("slic_fbank")
class SlicOcclusionFbankPerturbator(OcclusionFbankPerturbator):
    """
    Class for perturbing a given filterbank based on the segmentation obtained with the superpixel
    algorithm SLIC (https://ieeexplore.ieee.org/document/6205760). Different number of segments
    are used, thus different segmentations with different degrees of granularity are obtained.
    """
    def __init__(
            self,
            mask_probability: float,
            n_masks: int,
            segments_range: tuple,
            segments_step: int,
            slic_sigma: int,
            compactness: float = 0.03):
        self.mask_probability = mask_probability
        LOGGER.info(f"Masking fbanks with probability of {self.mask_probability}.")
        # Parameters to define granularity
        self.min_segments = segments_range[0]
        self.max_segments = segments_range[1]
        self.segments_step = segments_step
        # Adjust the number of masks according to make it multiple of the number of
        # segmentations to be performed
        self.n_segmentations = ((self.max_segments - self.min_segments) / self.segments_step) + 1
        self.n_masks_per_segmentation = n_masks // self.n_segmentations
        self.n_masks = self.n_masks_per_segmentation * self.n_segmentations
        LOGGER.info(
            f"{self.n_masks} masks are used, with {int(self.n_masks_per_segmentation)} "
            f"for each of the {int(self.n_segmentations)} segmentations.")
        # Parameters useful for SLIC. For more details, see
        # https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        self.slic_sigma = slic_sigma
        self.compactness = compactness
        # Cache for the segmentation
        self.test_index_to_segment = {}  # {test_index: {n_segments: np.array}}
        self.perturbation_stats = {}  # {entry: n_processed}

    @classmethod
    def from_config_dict(cls, config: Dict = None):
        if config is None:
            return cls(
                mask_probability=0.5,
                n_masks=8000,
                segments_range=(500, 1000),
                segments_step=100,
                slic_sigma=5,
                compactness=0.3)
        else:
            _config = config
            fbank_occlusion = _config.get("fbank_occlusion", {})
            n_masks = fbank_occlusion.get("n_masks", 8000)
            mask_probability = fbank_occlusion.get("p", 0.5)
            segments_range = fbank_occlusion.get("segments_range", (500, 1000))
            segments_step = fbank_occlusion.get("segments_step", 100)
            slic_sigma = fbank_occlusion.get("slic_sigma", 5)
            compactness = fbank_occlusion.get("compactness", 0.3)
            assert 0. < mask_probability <= 1.
            assert type(segments_range) == tuple
            return cls(
                mask_probability=mask_probability,
                n_masks=n_masks,
                segments_range=segments_range,
                segments_step=segments_step,
                slic_sigma=slic_sigma,
                compactness=compactness)

    def _do_segmentation(self, fbank: Tensor) -> Dict[int, np.array]:
        """
        Obtains the matrices with segmentations of different degrees of granularity.
        Args:
            - fbank (Tensor): the original filterbank.
        Returns:
            - Dict, containing different number of segments as keys, and the np.array
            segmentations as values.
        """
        min_value = torch.min(fbank)
        max_value = torch.max(fbank)
        normalized_fbank = (fbank - min_value) / (max_value - min_value)
        segment_dict = {}
        for n_segments in range(self.min_segments, self.max_segments + 1, self.segments_step):
            # The exact number of segments is not guaranteed, see
            # https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
            segment_dict[n_segments] = slic(
                normalized_fbank.numpy(),
                compactness=self.compactness,
                n_segments=n_segments,
                sigma=self.slic_sigma,
                channel_axis=None)
        return segment_dict

    def get_segments(self, fbank: Tensor, test_index: int, perturb_index: int) -> np.array:
        """
        Returns the segmentation for the given fbank if it is available,
        otherwise computes, stores it, and returns it.
        Finally, clean the cache if all the masks for a given test_index
        have been created.
        Args:
            - fbank (Tensor): the original filterbank
            - test_index (int): index of the original instance to be perturbed
            - perturb_index (int): index of the perturbed instance
        Return:
            - np.array containing the various segmentations with different n. of segments
        """
        if test_index not in self.test_index_to_segment.keys():
            self.test_index_to_segment[test_index] = self._do_segmentation(fbank)
        # Determine which segmentation the perturb_index corresponds to
        segmentation_group_index = (perturb_index % self.n_masks) // self.n_masks_per_segmentation
        n_segments = segmentation_group_index * self.segments_step + self.min_segments
        # Update statistics and eventually clean cache
        if test_index not in self.perturbation_stats.keys():
            self.perturbation_stats[test_index] = 0
        self.perturbation_stats[test_index] += 1
        segmentation = self.test_index_to_segment[test_index][n_segments]
        if not self.perturbation_stats[test_index] < self.n_masks:
            del self.test_index_to_segment[test_index]
        return segmentation

    def __call__(
            self,
            fbank: Tensor,
            test_index: int,
            perturb_index: int,
            *args,
            **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Performs perturbation on a given filterbank using a partition into segments.
        Args:
            - fbank (Tensor): the original filterbank to be manipulated
            - orig_index (int): index of the dataset obtained from the original test set
            - perturb_index (int): index of the new perturbed dataset
        Returns:
            - Tensor: the perturbed filterbank, of the same shape of fbank
        """
        segment_partition = self.get_segments(fbank, test_index, perturb_index)
        unique_values = torch.rand(
            segment_partition.max() + 1).ge(    # segment_partition range is from 1 to n_segments
            1 - self.mask_probability).long()
        mask = unique_values[segment_partition.flatten()].view(segment_partition.shape)
        masked_fbank = fbank * mask
        return mask, masked_fbank
