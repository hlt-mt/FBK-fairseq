#!/usr/bin/env python3 -u
# Copyright 2021 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import ast
import logging
import os
import sys
from argparse import Namespace
from typing import Optional

import numpy as np
import torch

from omegaconf import DictConfig
from torch import Tensor

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.models import FairseqEncoderDecoderModel


class EncoderStatesAverage:
    def __init__(self, model: FairseqEncoderDecoderModel):
        self.model = model
        self.total_sum_encoder_out: Optional[Tensor] = None
        self.num_encoder_outs = 0

    def add_batch(self, samples):
        with torch.no_grad():
            encoder_outs = self.model.encoder(**samples['net_input'])
            encoder_out = encoder_outs["encoder_out"][0]
            # This assures encoder outputs are zero in the padded vectors
            if len(encoder_outs["encoder_padding_mask"]) > 0:
                encoder_out = (encoder_out.transpose(2, 0) * ~encoder_outs["encoder_padding_mask"][0]).transpose(2, 0)
            batch_sum_encoder_outs = encoder_out.sum(dim=(0, 1))
            num_frames = encoder_out.shape[0] * encoder_out.shape[1]
            # If there is no padding mask, the whole input has to be considered
            if len(encoder_outs["encoder_padding_mask"]) > 0:
                num_frames -= encoder_outs["encoder_padding_mask"][0].sum()

            if self.total_sum_encoder_out is None:
                self.total_sum_encoder_out = batch_sum_encoder_outs
            else:
                self.total_sum_encoder_out += batch_sum_encoder_outs
            self.num_encoder_outs += num_frames

    @property
    def current_value(self):
        if self.total_sum_encoder_out is None:
            return None
        return (self.total_sum_encoder_out / self.num_encoder_outs).to(torch.device("cpu")).numpy()

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.current_value)


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def _main(cfg: DictConfig, output_file):
    """
    Similar to the generate.py, but it only computes the encoder outputs
    which are then averaged over the entire gen-subset.
    """
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate')

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
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    assert len(models) == 1 and isinstance(models[0], FairseqEncoderDecoderModel)
    model = models[0]

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    averager = EncoderStatesAverage(model)
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue
        averager.add_batch(sample)

    averager.save(cfg.task.save_file)


def cli_main():
    """
    Takes the same parameters of the generate.py, with the addition of the path where to save
    the tensor with the average of the encoder outputs of the given model on the given gen-subset.
    """
    parser = options.get_generation_parser()
    parser.add_argument("--save-file", required=True,
                        help="Path where to save the output numpy tensor")
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
