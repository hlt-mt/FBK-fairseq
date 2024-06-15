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
from typing import Tuple, Dict, List, Union

import numpy as np
import torch
from torch import Tensor
from skimage.segmentation import slic

from examples.speech_to_text.occlusion_explanation.perturbators import \
    register_perturbator, OcclusionFbankPerturbator


LOGGER = logging.getLogger(__name__)


class SlicOcclusionFbankPerturbatorBase(OcclusionFbankPerturbator):
    """
    Class for perturbing a given filterbank based on the segmentation obtained with the superpixel
    algorithm SLIC (https://ieeexplore.ieee.org/document/6205760). Different number of segments
    are used, thus different segmentations with different degrees of granularity are obtained.

    Concrete instances of this abstract class should implement:
     - _parse_custom_args: this method adds to the config dictionary the elements
     that are useful to the child class
     - get_n_segments: this method decides for each segment the number of segments to be
     used in the current segmentation level.
    """
    def __init__(
            self,
            mask_probability: float,
            n_masks: Union[int, List[int]],
            n_segments: list,
            slic_sigma: int,
            compactness: float):
        self.mask_probability = mask_probability
        LOGGER.info(f"Masking fbanks with probability of {self.mask_probability}.")
        # Parameters to define granularity
        self.n_segments = n_segments
        self.n_segmentations = len(n_segments)
        if isinstance(n_masks, int):
            # Adjust the number of masks to make it multiple of the number of segmentations to be performed
            self.n_masks_per_segmentation = n_masks // self.n_segmentations
            self.n_masks = self.n_masks_per_segmentation * self.n_segmentations
        elif isinstance(n_masks, list):
            assert len(n_masks) == self.n_segmentations, \
                "Each segmentation should have a number of masks associated."
            self.n_masks_per_segmentation = n_masks
            self.n_masks = sum(n_masks)
        # Parameters useful for SLIC. For more details, see
        # https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        self.slic_sigma = slic_sigma
        self.compactness = compactness
        # Cache for the segmentation
        self.test_index_to_segment = {}  # {test_index: {n_segments: np.array}}
        self.perturbation_stats = {}  # {entry: n_processed}

    def get_n_segments(self, n_frames: int, n_segments: int) -> int:
        """
        Returns the number of segments for a single sample at a single level of granularity.
        The exact number of segments, however, is not guaranteed in the output (see
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic).
        """
        raise NotImplementedError

    @staticmethod
    def _parse_custom_args(fbank_occlusion_config: Dict) -> Dict:
        """
        Adds arguments if needed in the derived classes.
        """
        return {}

    @classmethod
    def from_config_dict(cls, config: Dict = None):
        """
        Parsing of the config dictionary and initialization of class.
        The following parameters are configured:
            - n_masks (int or list): Number of masks applied to each filterbank. If a single value is provided,
            number of masks is equally divided over all the segmentation levels. Instead, if each level should
            have a different number of masks, n_masks should be a list with length equal to n_segments.
            Default is 8000.
            - mask_probability (float): Probability of each segment in the filterbank to be masked.
            Default is random.
            - n_segments (list): Number of segments to be obtained in each segmentation at different layers.
            Default is borrowed from the MFPP technique from which this approach is derived and is adapted to
            the average duration of MuST-C tst-COMMON (see https://arxiv.org/pdf/2006.02659.pdf).
            - slic_sigma (float): SLIC-specific parameter
            (see https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic).
            Default is borrowed from the MFPP technique.
            - compactness (float): SLIC-specific parameter
            (see https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic).
            Default is chosen based on manual inspection, providing good results.
        """
        fbank_occlusion = config.get("fbank_occlusion", {})
        class_args = {}
        class_args["n_masks"] = fbank_occlusion.get("n_masks", 8000)
        class_args["mask_probability"] = fbank_occlusion.get("p", 0.5)
        class_args["n_segments"] = fbank_occlusion.get("n_segments", [40, 80, 160, 320, 640])
        class_args["slic_sigma"] = fbank_occlusion.get("slic_sigma", 1)
        class_args["compactness"] = fbank_occlusion.get("compactness", 0.1)
        assert 0. <= class_args["mask_probability"] < 1.
        assert type(class_args["n_segments"]) == list
        # add arguments useful to the child class
        add_config = cls._parse_custom_args(fbank_occlusion)
        # merge class_args add_config
        class_args.update(add_config)
        return cls(**class_args)

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
        for granularity_level, n_segments in enumerate(self.n_segments):
            n_segments = self.get_n_segments(fbank.size(0), n_segments)
            LOGGER.debug(f"n_segments for filterbank of size {list(fbank.shape)}: {n_segments}")
            segment_dict[granularity_level] = slic(
                normalized_fbank.numpy(),
                compactness=self.compactness,
                n_segments=n_segments,
                sigma=self.slic_sigma,
                channel_axis=None)
            LOGGER.debug(f"n_segments returned by slic: {np.max(segment_dict[granularity_level])}")
        return segment_dict

    def get_granularity_level(self, perturb_index: int) -> int:
        """
        Returns which segmentation level the perturb_index corresponds to.
        """
        if isinstance(self.n_masks_per_segmentation, list):
            current_mask_idx = perturb_index % self.n_masks
            tot_mask = 0
            for i, m in enumerate(self.n_masks_per_segmentation):
                tot_mask += m
                if current_mask_idx < tot_mask:
                    return i
        elif isinstance(self.n_masks_per_segmentation, int):
            return perturb_index % self.n_masks // self.n_masks_per_segmentation
        else:
            raise ValueError(
                "n_masks_per_segmentation should be a list or int, "
                f"instead it is {type(self.n_masks_per_segmentation)}")

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
        # Add segmentation if not available
        if test_index not in self.test_index_to_segment.keys():
            self.test_index_to_segment[test_index] = self._do_segmentation(fbank)
        granularity_level = self.get_granularity_level(perturb_index)
        # Update statistics and eventually clean cache
        if test_index not in self.perturbation_stats.keys():
            self.perturbation_stats[test_index] = 0
        self.perturbation_stats[test_index] += 1
        segmentation = self.test_index_to_segment[test_index][granularity_level]
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
        # segment_partition range is from 1 to the number of segments in the current granularity level
        unique_values = torch.rand(
            segment_partition.max() + 1).ge(
            1 - self.mask_probability).long()
        mask = unique_values[segment_partition.flatten()].view(segment_partition.shape)
        masked_fbank = fbank * mask
        return mask, masked_fbank


@register_perturbator("slic_fbank_fixed_segments")
class SlicOcclusionFbankPerturbatorFixedSegments(SlicOcclusionFbankPerturbatorBase):
    """
    Class for perturbing a given filterbank based on the segmentation obtained with the superpixel
    algorithm SLIC. The number of segments across the various level of granularity is fixed regardless
    the duration of the audio sample.
    """
    def __init__(
            self,
            mask_probability: float,
            n_masks: Union[int, List[int]],
            n_segments: list,
            slic_sigma: int,
            compactness: float):
        super().__init__(mask_probability, n_masks, n_segments, slic_sigma, compactness)
        if isinstance(self.n_masks_per_segmentation, list):
            LOGGER.info(
                f"{self.n_masks} masks are used, with {self.n_masks_per_segmentation} masks "
                f"for the segmentations with {self.n_segments} segments.")
        else:
            LOGGER.info(
                f"{self.n_masks} masks are used, with {self.n_masks_per_segmentation} "
                f"for each of the {int(self.n_segmentations)} segmentations.")

    def get_n_segments(self, n_frames: int, n_segments: int) -> int:
        """
        Identity function as there is no adaptation based on the sample's duration.
        """
        return n_segments


@register_perturbator("slic_fbank_dynamic_segments")
class SlicOcclusionFbankPerturbatorDynamicSegments(SlicOcclusionFbankPerturbatorBase):
    """
    Class for perturbing a given filterbank based on the segmentation obtained with the superpixel
    algorithm SLIC. The number of segments across the various level of granularity is adapted based on
    the duration of the audio sample.
    """
    def __init__(
            self,
            mask_probability: float,
            n_masks: int,
            n_segments: list,
            slic_sigma: int,
            compactness: float,
            reference_duration: int):
        super().__init__(mask_probability, n_masks, n_segments, slic_sigma, compactness)
        self.reference_duration = reference_duration
        if isinstance(self.n_masks_per_segmentation, list):
            LOGGER.info(
                f"{self.n_masks} masks are used, with {self.n_masks_per_segmentation} masks "
                f"for the segmentations with {self.n_segments} segments considering a reference "
                f"duration of {(self.reference_duration * 10 + 25) / 1000} seconds.")
        else:
            LOGGER.info(
                f"{self.n_masks} masks are used, with {self.n_masks_per_segmentation} "
                f"for each of the {int(self.n_segmentations)} segmentations considering a reference "
                f"duration of {(self.reference_duration * 10 + 25) / 1000} seconds.")

    @staticmethod
    def _parse_custom_args(fbank_occlusion_config: Dict) -> Dict:
        """
        The following arguments are added:
            - reference_duration: represents the number of frames based on which the numbers of segments
            have been set. Default is 500, which represents the median number of frames of segments
            contained in MuST-C tst-COMMON
        """
        return {"reference_duration": fbank_occlusion_config.get("reference_duration", 500)}

    def get_n_segments(self, n_frames: int, n_segments: int) -> int:
        """
        Adjust the number of segments based on duration.
        """
        n_segments = round(n_segments * n_frames / self.reference_duration)
        # guarantee at least 1 segment and prevent division by 0 in slic
        n_segments = n_segments if n_segments != 0 else 1
        return n_segments
