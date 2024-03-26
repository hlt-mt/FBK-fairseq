#!/usr/bin/env python3 -u
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
"""
Generates and stores the saliency maps for a given test set and trained model using
occlusion-based techniques.
"""

from os import path as op
import ast
import logging
import sys
from argparse import Namespace
import random

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.occlusion_explanation.occlusion_transformer_decoder import \
    OcclusionTransformerDecoderScriptable
from examples.speech_to_text.occlusion_explanation.accumulator import Accumulator
from examples.speech_to_text.occlusion_explanation.configs import \
    PerturbConfig, add_occlusion_perturbation_args
from fairseq import options, utils, tasks, checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.models.speech_to_text import TransformerDecoderScriptable


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert cfg.task.save_file is not None, "--save-file required for saving."
    return _main(cfg, sys.stdout)


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file)
    logger = logging.getLogger('fairseq_cli.generate_occlusion_explanation')

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Set seed
    torch.manual_seed(cfg.common.seed)
    torch.cuda.manual_seed(cfg.common.seed)
    np.random.seed(cfg.common.seed)
    random.seed(cfg.common.seed)

    task = tasks.setup_task(cfg.task)
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    yaml_path = op.join(cfg.task.data, cfg.task.perturb_config)
    perturb_config = PerturbConfig(yaml_path)
    fbank_perturbator = perturb_config.get_perturbator_from_config("fbank_occlusion")
    decoder_perturbator = perturb_config.get_perturbator_from_config("decoder_occlusion")

    logger.info("Loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count)  # this includes task.build_model()

    assert len(models) == 1, "Ensemble of multiple models is not supported yet."
    model = models[0]

    occlusion_decoder = OcclusionTransformerDecoderScriptable(
        saved_cfg.model, tgt_dict, decoder_perturbator)
    if hasattr(model, "decoder"):
        assert isinstance(model.decoder, TransformerDecoderScriptable), \
            "Only Transformer decoder is currently supported."
        occlusion_decoder.load_state_dict(model.decoder.state_dict())
        model.decoder = occlusion_decoder
    else:
        raise ValueError("Only encoder-decoder models are currently supported.")

    if cfg.common.fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

    task.load_dataset(cfg.dataset.gen_subset)  # instantiating the dataset to be perturbed
    occluded_dataset = OccludedSpeechToTextDataset(  # creating the perturbed dataset
        task.dataset(cfg.dataset.gen_subset), fbank_perturbator, tgt_dict)

    scorer = perturb_config.get_scorer_from_config()

    accumulator = Accumulator(cfg.task, occluded_dataset)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=occluded_dataset,
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), model.max_positions()),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"))

    # Load original probabilities
    with h5py.File(cfg.task.original_probs, 'r') as f:
        orig_probs = {  # len(orig_probs) = n_samples,
            key: torch.tensor(f[key][()]) for key in f.keys()}  # each value has shape (padded_seq_len, dict_len))

    gen_timer = StopwatchMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        batch_size = sample["masks"].shape[0]
        num_entries = torch.unique(sample["orig_id"]).shape[0]

        gen_timer.start()
        with torch.no_grad():
            decoder_out, ctc_outputs = model(
                sample["net_input"]["src_tokens"],
                sample["net_input"]["src_lengths"],
                sample["net_input"]["target"])
            tgt_embed_masks = decoder_out[1]["masks"]
            perturb_probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
                decoder_out, log_probs=False)
            assert torch.all((perturb_probs >= 0) & (perturb_probs <= 1))

        single_fbank_heatmaps, fbank_masks, single_decoder_heatmaps, decoder_masks = scorer(
            sample=sample,
            orig_probs=orig_probs,
            perturb_probs=perturb_probs,
            tgt_embed_masks=tgt_embed_masks)

        gen_timer.stop(batch_size)
        logger.info(f"Generated {batch_size} single heatmaps for {num_entries} entries")

        accumulator(sample, single_fbank_heatmaps, fbank_masks, single_decoder_heatmaps, decoder_masks)


def cli_main():
    parser = options.get_generation_parser()
    add_occlusion_perturbation_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
