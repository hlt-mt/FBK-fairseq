#!/usr/bin/env python3 -u
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
"""
Generate random explanations for a given dataset and save them in a h5 file.
Random explanations are 3D Tensors (num_tokens, time/num_tokens, channels/tgt_embed_dim/1)
containing values that follow a normal distribution with 0 mean and 1 variance.
"""
import logging
import os
from typing import Dict, Tuple

import h5py
from tqdm import tqdm

import torch
from torch import Tensor

from fairseq import tasks, options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


def generate_random_explanations(
        item: Tuple, tgt_dim: int, fbank_channels: int = None, seed: int = None) -> Dict[str, Tensor]:
    if seed is not None:
        torch.manual_seed(seed)

    source = item[1]
    target = item[2]

    if fbank_channels is None:
        fbank_channels = source.size(1)

    random_fbank_heatmap = torch.randn(target.size(0), source.size(0), fbank_channels)
    random_tgt_embed_heatmap = torch.randn(target.size(0), target.size(0), tgt_dim)
    return {
        "fbank_heatmap": random_fbank_heatmap,
        "tgt_embed_heatmap": random_tgt_embed_heatmap}


def save_explanations(
        index: int, file_path: str, random_explanations: Dict[str, Tensor]) -> None:
    with h5py.File(file_path, "a") as f:
        group = f.create_group(str(index))
        # Iterate over dictionary keys and save data as datasets
        for key, value in random_explanations.items():
            group.create_dataset(key, data=value)


def main(args):
    cfg = convert_namespace_to_omegaconf(args)
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger('fairseq_cli.generate_random_explanations')

    logger.info(f"Set seed: {cfg.common.seed}")

    # load task and dataset
    task = tasks.setup_task(cfg.task)
    task.load_dataset(cfg.dataset.gen_subset)
    ds = task.dataset(cfg.dataset.gen_subset)

    # generate and save random explanations
    save_file_hdf5 = args.output_file + ".h5"
    assert not os.path.exists(save_file_hdf5), f"The file {save_file_hdf5} already exists."

    fbank_channels = 1 if args.src_discrete else None
    tgt_dim = 1 if args.tgt_discrete else args.decoder_embed_dim

    for i in tqdm(range(len(ds))):
        random_explanations = generate_random_explanations(
            ds[i], tgt_dim, fbank_channels, cfg.common.seed)
        save_explanations(i, save_file_hdf5, random_explanations)


if __name__ == "__main__":
    parser = options.get_generation_parser()
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to the h5 file where random heatmaps are saved.")
    parser.add_argument(
        "--src-discrete",
        action="store_true",
        help="If specified, random discrete explanations will be generated for the filterbank, "
             "otherwise explanations will fit the channel dimension.")
    parser.add_argument(
        "--tgt-discrete",
        action="store_true",
        help="If specified, random discrete explanations will be generated for the previous "
             "output tokens, otherwise explanations will fit the target embedding dimension.")
    parser.add_argument(
        "--decoder-embed-dim",
        typ=int,
        default=512,
        help="Embedding dimension of previous output tokens. Default is 512.")
    args = options.parse_args_and_arch(parser)
    main(args)
