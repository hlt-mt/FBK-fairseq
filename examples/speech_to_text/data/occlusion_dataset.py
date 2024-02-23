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

from typing import List, Tuple, Dict, Optional, Union
import logging

import numpy as np
import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.perturbators.encoder_perturbator import OcclusionFbankPerturbator
from fairseq.data import ConcatDataset, BaseWrapperDataset, Dictionary
from fairseq.data.audio.speech_to_text_dataset import _collate_frames, SpeechToTextDataset
from fairseq.data import data_utils


LOGGER = logging.getLogger(__name__)


class OccludedSpeechToTextDataset(BaseWrapperDataset):
    """
    Wrapper for Dataset class to perturb data.
    """
    def __init__(
            self,
            to_be_occluded_dataset: ConcatDataset,
            perturbator: OcclusionFbankPerturbator,
            tgt_dict: Dictionary):
        super().__init__(to_be_occluded_dataset)
        for d in self.dataset.datasets:
            assert issubclass(type(d), SpeechToTextDataset), \
                "Datasets must be an instance of SpeechToTextDataset."
        self.perturbator = perturbator
        self.tgt_dict = tgt_dict
        self.n_masks = self.perturbator.n_masks

    def _original_index(self, perturb_index: int) -> int:
        """
        Returns the index of the original dataset mapped to the
        index of the perturbed dataset
        """
        return perturb_index // self.n_masks

    def __len__(self):
        return len(self.dataset) * self.n_masks

    def __getitem__(self, perturb_index: int) -> Tuple[int, int, Tensor, Tensor, Tensor, Optional[Tensor]]:
        orig_dataset_index = self._original_index(perturb_index)
        returned_tuple = self.dataset[orig_dataset_index]
        if len(returned_tuple) == 3:
            index, fbank, predicted_tokens = returned_tuple
            source_text = None
        else:
            index, fbank, predicted_tokens, source_text = returned_tuple
        mask, perturbed_fbank = self.perturbator(fbank, orig_dataset_index, perturb_index)
        return perturb_index, orig_dataset_index, mask, \
               perturbed_fbank, predicted_tokens, source_text,

    def collater(self, samples: List[Tuple[int, int, Tensor, Tensor, Tensor, Optional[Tensor]]]) -> Dict:
        if len(samples) == 0:
            return {}
        perturb_indices = torch.tensor([i for i, _, _, _, _, _ in samples], dtype=torch.long)
        orig_indices = torch.tensor([i for _, i, _, _, _, _ in samples], dtype=torch.long)
        masks = _collate_frames([m for _, _, m, _, _, _ in samples])
        frames = _collate_frames([s for _, _, _, s, _, _ in samples])
        n_frames = torch.tensor([s.size(0) for _, _, _, s, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        perturb_indices = perturb_indices.index_select(0, order)
        orig_indices = orig_indices.index_select(0, order)
        frames = frames.index_select(0, order)
        masks = masks.index_select(0, order)
        predicted_tokens = data_utils.collate_tokens(
            [p for _, _, _, _, p, _ in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True)
        predicted_tokens = predicted_tokens.index_select(0, order)
        predicted_lengths = torch.tensor(
            [p.size(0) for _, _, _, _, p, _ in samples], dtype=torch.long).index_select(0, order)
        src_texts = [s for _, _, _, _, _, s in samples]
        src_texts = [src_texts[i] for i in order]
        out = {
            "id": perturb_indices,
            "orig_id": orig_indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "target": predicted_tokens,
                "target_lengths": predicted_lengths},
            "masks": masks,
            "src_texts": src_texts,
            "nsentences": len(samples)}
        return out

    @property
    def sizes(self):
        base_sizes = super().sizes
        occluded_sizes = []
        for s in base_sizes:
            occluded_sizes.extend([s for _ in range(self.n_masks)])
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
        # Define ordered indices for the occlusion dataset starting from the ordered indices of
        # the original dataset
        for value in base_order_indices:
            occluded_order_indices.extend(value * self.n_masks + n for n in range(self.n_masks))
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
            np.array: filtered sample array
            list: list of removed indices
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
