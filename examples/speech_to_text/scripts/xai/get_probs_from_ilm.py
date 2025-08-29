#!/usr/bin/env python3 -u
# Copyright 2025 FBK

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
To simulate a model's internal language model (ILM), do constrained generation using only
the decoder of a trained model (instead of the encoder pass, a dummy encoder output is used)
and save probabilities. This script was designed to study the translation of gender terms
and saves probabilities of the generated hypotheses and a swapped version with a different gender.
"""

import ast
import logging
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from examples.speech_to_text.tasks.speech_to_text_ctc_genderxai import SpeechToTextGenderXaiCtcTask
from examples.speech_to_text.tasks.speech_to_text_genderxai import SpeechToTextGenderXaiTask
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
    assert cfg.task.dummy_encoder_outs is not None, \
        "--dummy-encoder-outs required for generation with ILM!"
    return _main(cfg)


def _main(cfg: DictConfig):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stderr)
    logger = logging.getLogger("fairseq_cli.get_probs_from_ilm")

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
    assert isinstance(task, SpeechToTextGenderXaiTask) or \
        isinstance(task, SpeechToTextGenderXaiCtcTask), \
        "This script is intended for the study of gender terms " \
        "and assumes the presence of hypotheses with swapped gender in the dataset."

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

    assert len(models) == 1, "Ensemble of multiple models is not supported yet."
    model = models[0]

    # loading the dataset should happen after the checkpoint has been loaded so we can give it
    # the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

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
        max_positions=utils.resolve_max_positions(task.max_positions(), model.max_positions()),
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

    with open(cfg.task.dummy_encoder_outs, 'rb') as f:
        dummy_encoder_outs = torch.from_numpy(np.load(f))
    dummy_encoder_outs = dummy_encoder_outs.to(next(model.parameters()).dtype)
    dummy_encoder_outs = utils.move_to_cuda(dummy_encoder_outs) if use_cuda else dummy_encoder_outs

    for sample in progress:
        # Perform the forward pass with forced decoding and save the output probabilities
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        batch_size, max_length, num_channels = sample["net_input"]["src_tokens"].shape  # (B, T, C)
        shaped_encoder_outs = dummy_encoder_outs.repeat(1, batch_size, 1)  # Make it of shape (T, B, C)

        if "net_input" not in sample:
            continue

        # Generated hypothesis
        ilm_out = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out={"encoder_out": [shaped_encoder_outs], "encoder_padding_mask": []})
        ilm_probs = model.get_normalized_probs(ilm_out, log_probs=False, sample=None)  # (B, S, V)
        assert torch.all((ilm_probs >= 0) & (ilm_probs <= 1))

        # Swapped hypothesis
        ilm_out = model.decoder(
            prev_output_tokens=sample["net_input"]["swapped_prev_output_tokens"],
            encoder_out={"encoder_out": [shaped_encoder_outs], "encoder_padding_mask": []})
        ilm_swapped_probs = model.get_normalized_probs(
            ilm_out, log_probs=False, sample=None)  # (B, S, V)
        assert torch.all((ilm_swapped_probs >= 0) & (ilm_swapped_probs <= 1))

        with h5py.File(cfg.task.save_file, "a") as f:
            for i, sample_id in enumerate(sample["id"].tolist()):
                ilm_probs_wo_padding = ilm_probs[i, :sample["target_lengths"][i], :].cpu().detach().numpy()
                ilm_swapped_probs_wo_padding = ilm_swapped_probs[i, :sample["swapped_target_lengths"][i], :].cpu().detach().numpy()
                f.create_dataset(
                    str(sample_id),
                    data=ilm_probs_wo_padding)
                f.create_dataset(
                    str(sample_id) + "_swapped",
                    data=ilm_swapped_probs_wo_padding)


@dataclass
class SavingConfig(FairseqDataclass):
    save_file: str = field(
        default=None,
        metadata={"help": "Path to an h5 file where probabilities will be saved."})
    dummy_encoder_outs: str = field(
        default=None,
        metadata={"help": "Path to a numpy file containing the matrix to use as a dummy encoder output."})


def add_saving_args(parser):
    group = parser.add_argument_group("saving")
    gen_parser_from_dataclass(group, SavingConfig())
    return group


def cli_main():
    """
    This script expects the usual arguments for generation, plus:
    --save-file: path to an h5 file where probabilities will be saved.
    --dummy-encoder-outs: 
        path to a numpy file containing the vector to use as a dummy encoder output,
        this should be a 1D array of size equal to the encoder output dimension for the chosen model.
    N.B.: Since this script performs forced decoding on the target text, it's important that the 
    model's yaml configuration does not contain the bpe_tokenizer field, and that the target text 
    is already tokenized in the dataset. 
    """
    parser = options.get_generation_parser()
    add_saving_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
