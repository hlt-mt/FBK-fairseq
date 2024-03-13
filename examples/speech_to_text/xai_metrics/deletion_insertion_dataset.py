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

from typing import List, Tuple, Dict, Union

import numpy as np
import torch
from torch import Tensor

from fairseq.data import ConcatDataset, BaseWrapperDataset
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fairseq.data import data_utils


class FeatureAttributionEvaluationSpeechToTextDataset(BaseWrapperDataset):
    """
    Wrapper for the Dataset class designed to perturb data for Deletion and Insertion metrics.
    These metrics serve to evaluate explanations by quantifying performance variations through
    the deletion or insertion of the most relevant features as identified by the explanation.
    """
    def __init__(
            self,
            original_dataset: ConcatDataset,
            aggregated_fbank_heatmaps: Dict[int, Tuple[Tensor, Tensor]],
            interval_size: int,
            num_intervals: int,
            metric: str):
        super().__init__(original_dataset)
        assert issubclass(type(self.dataset.datasets[0]), SpeechToTextDataset), \
            "Dataset must be an instance of SpeechToTextDataset."
        assert len(self.dataset.datasets) == 1, "Only one dataset at a time supported."
        assert len(self.dataset.datasets[0]) == len(aggregated_fbank_heatmaps), \
            "Number of explanations does not match the number of samples in the dataset."
        self.aggregated_fbank_heatmaps = aggregated_fbank_heatmaps
        self.interval_size = interval_size
        self.num_intervals = num_intervals
        assert metric == "deletion" or metric == "insertion", \
            "Incorrect metric type. Accepted types are ['deletion' or 'insertion']"
        self.metric = metric
        self.id_to_ordered_indices = {}

    def _original_index(self, perturb_index: int) -> int:
        """
        Returns the index of the original dataset mapped to the
        index of the perturbed dataset used for the metric.
        """
        return perturb_index // self.num_intervals

    def __len__(self):
        return len(self.dataset) * self.num_intervals

    def set_id_to_ordered_indices(self, orig_index: int) -> None:
        """
        Updates the 'id_to_ordered_indices' dictionary by associating each heatmap ID with
        the corresponding array of ordered indices used for deletion and insertion operations.
        """
        ordered_indices = torch.argsort(self.aggregated_fbank_heatmaps[orig_index][0].flatten())
        self.id_to_ordered_indices[orig_index] = ordered_indices

    def __getitem__(
            self,
            perturb_index: int) -> Union[Tuple[int, Tensor, Tensor], Tuple[int, Tensor, Tensor, Tensor]]:
        orig_dataset_index = self._original_index(perturb_index)
        returned_tuple = self.dataset[orig_dataset_index]
        fbank = returned_tuple[1]
        if orig_dataset_index not in self.id_to_ordered_indices:
            self.set_id_to_ordered_indices(orig_dataset_index)

        # get the percentage value of elements to be inserted/removed based on the perturb_index
        perc_value = (perturb_index % self.num_intervals) * self.interval_size
        # make the percentage value a real number based on the size of the explanation
        n_elements = fbank.numel() * perc_value // 100
        removal_index = fbank.numel() - n_elements

        ordered_indices = self.id_to_ordered_indices[orig_dataset_index]
        fbank_size = fbank.size()
        flatten_fbank = fbank.flatten()
        if self.metric == "deletion":
            flatten_fbank[ordered_indices[removal_index:]] = 0
        else:
            flatten_fbank[ordered_indices[:removal_index]] = 0
        fbank = flatten_fbank.view(fbank_size)
        returned_list = list(returned_tuple)
        returned_list[1] = fbank
        returned_list[0] = perturb_index
        return tuple(returned_list)

    def collater(
            self,
            samples: List[Union[Tuple[int, Tensor, Tensor], Tuple[int, Tensor, Tensor, Tensor]]]) -> Dict:
        out = super().collater(samples)
        if out:
            out["orig_id"] = torch.tensor([self._original_index(i) for i in out["id"]])
        return out

    @property
    def sizes(self):
        base_sizes = super().sizes
        occluded_sizes = []
        for s in base_sizes:
            occluded_sizes.extend([s for _ in range(self.num_intervals)])
        return np.array(occluded_sizes)

    def num_tokens(self, index):
        original_index = self._original_index(index)
        return self.dataset.num_tokens(original_index)

    def size(self, index):
        original_index = self._original_index(index)
        return self.dataset.size(original_index)

    def ordered_indices(self):
        base_order_indices = super().ordered_indices()
        occluded_order_indices = []
        # Define ordered indices for the perturbed dataset used to compute the metric
        # starting from the ordered indices of the original dataset
        for value in base_order_indices:
            occluded_order_indices.extend(value * self.num_intervals + n for n in range(self.num_intervals))
        return np.array(occluded_order_indices)

    def attr(self, attr: str, index: int):
        original_index = self._original_index(index)
        return self.dataset.attr(attr, original_index)

    def batch_by_size(
            self,
            indices,
            max_tokens=None,
            max_sentences=None,
            required_batch_size_multiple=1):
        return data_utils.batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=None)

    def filter_indices_by_size(
            self, indices: np.array, max_sizes: Union[int, List[int], Tuple[int]]) -> Tuple[np.array, List]:
        """
        Returns:
            - np.array: filtered sample array
            - list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif hasattr(self, "sizes") and isinstance(self.sizes, list) and len(self.sizes) == 1:
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(indices, self.size, max_sizes)
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(indices, self.size, max_sizes)
        return indices, ignored
