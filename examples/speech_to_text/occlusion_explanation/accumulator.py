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
        self.save_file = args.save_file
        self.n_masks = perturbation_dataset.n_masks
        self.final_masks = {}
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
        Iterates over the batch in the collated_data and gets heatmaps
        (stripped of padding) for the same items.
        """
        orig_indices: LongTensor = collated_data["orig_id"]
        for i, orig_ind in enumerate(orig_indices):
            ind = orig_ind.item()
            # Strip padding
            src_length = collated_data["net_input"]["src_lengths"][i].item()
            target_length = collated_data["net_input"]["target_lengths"][i].item()
            current_fbank_heatmap = fbank_heatmaps[i][:target_length, :src_length]
            current_tgt_embed_heatmap = tgt_embed_heatmaps[i][:target_length, :target_length]
            current_fbank_mask = fbank_masks[i][:target_length, :src_length]
            current_tgt_embed_mask = tgt_embed_masks[i][:target_length, :target_length]
            if ind in self.final_masks.keys():
                self.final_masks[ind]["n_masks"] += 1
                self.final_masks[ind]["fbank_heatmap"] += current_fbank_heatmap
                self.final_masks[ind]["tgt_embed_heatmap"] += current_tgt_embed_heatmap
                self.final_masks[ind]["fbank_mask"] += current_fbank_mask
                self.final_masks[ind]["tgt_embed_mask"] += current_tgt_embed_mask
            else:
                self.final_masks[ind] = {
                    "src_text": decode_line(
                        collated_data["src_texts"][i],
                        self.src_dict,
                        self.bpe_tokenizer_src),
                    "tgt_text": decode_line(
                        collated_data["net_input"]["target"][i],
                        self.tgt_dict),
                    "n_masks": 1,
                    "fbank_heatmap": current_fbank_heatmap,
                    "tgt_embed_heatmap": current_tgt_embed_heatmap,
                    "fbank_mask": current_fbank_mask,
                    "tgt_embed_mask": current_tgt_embed_mask}

    def _normalize(self, index: int) -> None:
        """
        Gets the heatmaps divided by the sum of all the heatmaps.
        """
        self.final_masks[index]["fbank_heatmap"] = torch.where(
            self.final_masks[index]["fbank_mask"] != 0,
            self.final_masks[index]["fbank_heatmap"] / self.final_masks[index]["fbank_mask"],
            torch.tensor(0., device=self.final_masks[index]["fbank_mask"].device))
        self.final_masks[index]["tgt_embed_heatmap"] = torch.where(
            self.final_masks[index]["tgt_embed_mask"] != 0,
            self.final_masks[index]["tgt_embed_heatmap"] / self.final_masks[index]["tgt_embed_mask"],
            torch.tensor(0., device=self.final_masks[index]["fbank_mask"].device))

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

        # Create an HDF5 file to store accumulated data
        save_file_hdf5 = self.save_file + ".h5"
        with h5py.File(save_file_hdf5, "a") as f:
            for orig_ind in list(self.final_masks):
                if self.final_masks[orig_ind]["n_masks"] == self.n_masks or \
                        self.final_masks[orig_ind]["n_masks"] > self.n_masks:
                    self._normalize(orig_ind)
                    del self.final_masks[orig_ind]["fbank_mask"]
                    del self.final_masks[orig_ind]["tgt_embed_mask"]
                    # Create a group for each orig_ind
                    group = f.create_group(str(orig_ind))
                    # Iterate over dictionary keys and save data as datasets
                    for key, value in self.final_masks[orig_ind].items():
                        group.create_dataset(
                            key, data=value.cpu() if type(value) == Tensor else value)
                    LOGGER.info(f"Attribution maps for sample {orig_ind} have been saved.")

                    del self.final_masks[orig_ind]  # cleaning cache
