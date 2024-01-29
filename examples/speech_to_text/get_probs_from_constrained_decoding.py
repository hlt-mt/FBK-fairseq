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
Do constrained generation and save probabilities with a trained model.
"""

import ast
import logging
import os
import sys
from argparse import Namespace
from itertools import chain
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, gen_parser_from_dataclass
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert cfg.task.save_file is not None, "--save-file required for saving!"
    save_file_hdf5 = cfg.task.save_file + ".h5"
    return _main(cfg, sys.stdout, save_file_hdf5)


def _main(cfg: DictConfig, output_file, save_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file)
    logger = logging.getLogger("fairseq_cli.get_probs_from_consrained_decoding")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count)

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    assert len(models) == 1, "Ensemble of multiple models is not supported yet."
    model = models[0]

    if cfg.common.fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(task.max_positions(), * model.max_positions()),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"))

    gen_save_timer = StopwatchMeter()

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        gen_save_timer.start()
        if "net_input" not in sample:
            continue
        decoder_out, _ = model(**sample["net_input"])
        probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
            decoder_out, log_probs=False)
        assert torch.all((probs >= 0) & (probs <= 1))
        # probs = decoder_out[0]
        with h5py.File(save_file, "a") as f:
            for i, sample_id in enumerate(sample["id"].tolist()):
                f.create_dataset(
                    str(sample_id),
                    data=probs[i, :sample["target_lengths"][i], :].cpu().detach().numpy())  # strip padding

        gen_save_timer.stop(len(sample))


@dataclass
class SavingConfig(FairseqDataclass):
    save_file: str = field(default=None, metadata={"help": "File where probabilities will be saved."})


def add_saving_args(parser):
    group = parser.add_argument_group("saving")
    # fmt: off
    gen_parser_from_dataclass(group, SavingConfig())
    # fmt: on
    return group


def cli_main():
    parser = options.get_generation_parser()
    add_saving_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
