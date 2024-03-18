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
from itertools import compress
from typing import Dict, Union, List

import h5py
import torch
from torch import Tensor, LongTensor

from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from fairseq.data import Dictionary


LOGGER = logging.getLogger(__name__)


def decode_line(
        line: LongTensor,
        dictionary: Dictionary = None,
        bpe_tokenizer=None) -> Union[List[str], str, LongTensor]:
    if dictionary is None:
        return line
    decoded_sentence = [dictionary[ind] for ind in line if ind != dictionary.pad_index]
    if bpe_tokenizer is not None:
        decoded_sentence = bpe_tokenizer.decode(" ".join(decoded_sentence))
    return decoded_sentence


class Accumulator:
    """
    Accumulates the single heatmaps for each sample, performs saving, and cleans the cache.
    """
    def __init__(
            self,
            args,
            perturbation_dataset: OccludedSpeechToTextDataset):
        self.save_file_hdf5 = args.save_file + ".h5"
        self.n_masks = perturbation_dataset.n_masks
        self.accumulated_masks = {}
        self.tgt_dict = perturbation_dataset.tgt_dict
        for d in perturbation_dataset.dataset.datasets:
            if issubclass(type(d), SpeechToTextDatasetWithSrc):
                self.src_dict = d.src_dict
                self.bpe_tokenizer_src = d.bpe_tokenizer_src
                break
            else:
                self.src_dict = None
                self.bpe_tokenizer_src = None

    def _update_heatmaps(
            self,
            collated_data: Dict,
            fbank_heatmaps: Tensor,
            fbank_masks: Tensor,
            tgt_embed_heatmaps: Tensor,
            tgt_embed_masks) -> None:
        """
        Update heatmaps for multiple samples in batch.
        """
        orig_indices = collated_data["orig_id"]
        unique_indices, counts = torch.unique(orig_indices, return_counts=True)

        for orig_ind, count in zip(unique_indices, counts):
            ind = orig_ind.item()
            n_masks = count.item()
            idx_mask = orig_indices == orig_ind

            src_text = decode_line(
                list(compress(collated_data["src_texts"], idx_mask))[0],
                self.src_dict,
                self.bpe_tokenizer_src)
            tgt_text = decode_line(
                collated_data["net_input"]["target"][idx_mask][0],
                self.tgt_dict)
            src_length = collated_data["net_input"]["src_lengths"][idx_mask][0].item()
            target_length = collated_data["net_input"]["target_lengths"][idx_mask][0].item()

            # Strip padding and summation
            fbank_heatmap_sum = fbank_heatmaps[idx_mask, :target_length, :src_length].sum(dim=0)
            tgt_embed_heatmap_sum = tgt_embed_heatmaps[idx_mask, :target_length, :target_length].sum(dim=0)
            fbank_mask_sum = fbank_masks[idx_mask, :target_length, :src_length].sum(dim=0)
            tgt_embed_mask_sum = tgt_embed_masks[idx_mask, :target_length, :target_length].sum(dim=0)

            if ind in self.accumulated_masks:
                self.accumulated_masks[ind]["n_masks"] += n_masks
                self.accumulated_masks[ind]["fbank_heatmap"] += fbank_heatmap_sum
                self.accumulated_masks[ind]["tgt_embed_heatmap"] += tgt_embed_heatmap_sum
                self.accumulated_masks[ind]["fbank_mask"] += fbank_mask_sum
                self.accumulated_masks[ind]["tgt_embed_mask"] += tgt_embed_mask_sum
            else:
                self.accumulated_masks[ind] = {
                    "src_text": src_text,
                    "tgt_text": tgt_text,
                    "n_masks": n_masks,
                    "fbank_heatmap": fbank_heatmap_sum,
                    "tgt_embed_heatmap": tgt_embed_heatmap_sum,
                    "fbank_mask": fbank_mask_sum,
                    "tgt_embed_mask": tgt_embed_mask_sum}

    def _normalize(self, index: int) -> None:
        """
        Gets the heatmaps divided by the sum of all the heatmaps.
        """
        self.accumulated_masks[index]["fbank_heatmap"] = torch.where(
            self.accumulated_masks[index]["fbank_mask"] != 0,
            self.accumulated_masks[index]["fbank_heatmap"] / self.accumulated_masks[index]["fbank_mask"],
            torch.tensor(0., device=self.accumulated_masks[index]["fbank_mask"].device))
        self.accumulated_masks[index]["tgt_embed_heatmap"] = torch.where(
            self.accumulated_masks[index]["tgt_embed_mask"] != 0,
            self.accumulated_masks[index]["tgt_embed_heatmap"] / self.accumulated_masks[index]["tgt_embed_mask"],
            torch.tensor(0., device=self.accumulated_masks[index]["fbank_mask"].device))

    def __call__(
            self,
            samples: Dict,
            fbank_heatmaps: Tensor,
            fbank_masks: Tensor,
            target_embedding_heatmaps: Tensor,
            tgt_embed_masks) -> None:
        """
        Accumulates the single heatmaps to obtain the final attribution maps,
        then saves them. Accumulation consists in a sort of normalization.
        Args:
            - sample: output of the collater;
            - fbank_heatmaps: Tensor: single heatmaps related to each filterbank of the
            batch; the shape is (batch_size, sequence length, time, channels) or
            (batch_size, sequence length, time/channels, 1) depending on the type of perturbation;
            - tgt_embed_heatmaps: Tensor: single heatmaps related to the embeddings of each
            token; the shape is (batch_size, sequence length, sequence length, embedding dimension/1);
        Returns:
            - None
        """
        self._update_heatmaps(
            samples, fbank_heatmaps, fbank_masks, target_embedding_heatmaps, tgt_embed_masks)

        unique_ids = torch.unique(samples["orig_id"]).tolist()
        finalized_ids = [
            orig_id for orig_id in unique_ids if self.accumulated_masks[orig_id]["n_masks"] >= self.n_masks]
        if len(finalized_ids) > 0:
            # Create an HDF5 file to store accumulated data
            with h5py.File(self.save_file_hdf5, "a") as f:
                for orig_id in finalized_ids:
                    self._normalize(orig_id)
                    del self.accumulated_masks[orig_id]["fbank_mask"]
                    del self.accumulated_masks[orig_id]["tgt_embed_mask"]
                    # Create a group for each orig_ind
                    group = f.create_group(str(orig_id))
                    # Iterate over dictionary keys and save data as datasets
                    for key, value in self.accumulated_masks[orig_id].items():
                        group.create_dataset(
                            key, data=value.cpu() if type(value) == Tensor else value)
                    LOGGER.info(f"Attribution maps for sample {orig_id} have been saved.")

                    del self.accumulated_masks[orig_id]  # cleaning cache
