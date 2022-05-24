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

"""
Translate pre-processed data with a trained model.
"""
import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain
from typing import Optional

import numpy as np
import torch
from dataclasses import field, dataclass

from omegaconf import DictConfig

from examples.speech_to_text.utils.tags import join_tags_tokens
from fairseq import scoring, checkpoint_utils, options, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, gen_parser_from_dataclass
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq_cli.generate import get_symbols_to_strip_from_output


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


    # Set dictionaries
    src_dict = getattr(task, 'source_dictionary', None)
    tgt_dict = task.target_dictionary

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

    def load_lm(lm_path):
        """
        Helper function to load language models
        """
        lm = None
        if lm_path is not None:
            overrides["data"] = cfg.task.data

            try:
                lms, _ = checkpoint_utils.load_model_ensemble(
                    [lm_path], arg_overrides=overrides, task=None
                )
            except:
                logger.warning(
                    f"Failed to load language model! Please make sure that the language model dict is the same "
                    f"as target dict and is located in the data dir ({cfg.task.data})"
                )
                raise

            assert len(lms) == 1
            lm = lms[0]
        return lm

    primary_lm_path = getattr(cfg, "primary_lm_path", None)
    auxiliary_lm_path = getattr(cfg, "primary_lm_path", None)
    primary_lm_path = getattr(cfg.task, "primary_lm_path", None) if primary_lm_path is None else primary_lm_path
    auxiliary_lm_path = getattr(cfg.task, "primary_lm_path", None) if auxiliary_lm_path is None else auxiliary_lm_path

    primary_lm = load_lm(primary_lm_path)
    auxiliary_lm = load_lm(auxiliary_lm_path)

    # Optimize ensemble for generation
    for model in chain(models, [primary_lm, auxiliary_lm]):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

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

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(
        models, cfg.generation
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)
    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        gen_timer.start()
        hypos = task.inference_step(generator, models, sample, prefix_tokens)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(sample_id)
            else:
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            if has_target:
                target_str = decode_fn(target_str)

            if not cfg.common_eval.quiet:
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:cfg.generation.nbest]):
                hypo_tokens = hypo['tokens'].int().cpu()
                hypo_str = tgt_dict.string(
                    hypo_tokens,
                    cfg.common_eval.post_process,
                    escape_unk=True,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                        generator
                    ),
                )
                detok_hypo_str = decode_fn(hypo_str)
                hypo_aux_tokens = hypo['aux_tokens'].int().cpu()
                hypo_aux_str = src_dict.string(
                    hypo_aux_tokens,
                    cfg.common_eval.post_process,
                    extra_symbols_to_ignore={
                        generator.src_eos,
                    })
                detok_hypo_aux_str = decode_fn(hypo_aux_str)
                if not cfg.common_eval.quiet:
                    score = hypo['score'] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                    # detokenized hypothesis
                    print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)
                    print('AUX-{}\t{}\t{}'.format(sample_id, score, hypo_aux_str), file=output_file)
                    # detokenized hypothesis
                    print('AUXD-{}\t{}\t{}'.format(sample_id, score, detok_hypo_aux_str), file=output_file)

                    if "tags" in hypo and hypo["tags"] is not None:
                        tags_strings, joint_string = join_tags_tokens(
                            hypo["tags"].int().cpu(), hypo_tokens, tgt_dict, task.data_cfg.tags)

                        print('TAGS-{}\t{}\t{}'.format(sample_id, score, " ".join(tags_strings)), file=output_file)
                        hypo_joint_str = tgt_dict.string(
                            joint_string,
                            cfg.common_eval.post_process,
                            escape_unk=True,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                                generator
                            ),
                        )
                        detok_hypo_joint_str = decode_fn(hypo_joint_str)
                        print('JOINTD-{}\t{}\t{}'.format(sample_id, score, detok_hypo_joint_str), file=output_file)
                    if "aux_tags" in hypo and hypo["aux_tags"] is not None:
                        tags_strings, joint_string = join_tags_tokens(
                            hypo["aux_tags"].int().cpu(), hypo_aux_tokens, src_dict, task.data_cfg.tags)
                        print('AUXTAGS-{}\t{}\t{}'.format(sample_id, score, " ".join(tags_strings)), file=output_file)
                        hypo_joint_str = src_dict.string(
                            joint_string,
                            cfg.common_eval.post_process,
                            escape_unk=True,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                                generator
                            ),
                        )
                        detok_hypo_joint_str = decode_fn(hypo_joint_str)
                        print('AUXJOINTD-{}\t{}\t{}'.format(sample_id, score, detok_hypo_joint_str), file=output_file)

                    if cfg.generation.print_step:
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or cfg.common_eval.post_process is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        hypo_tokens = tgt_dict.encode_line(detok_hypo_str, add_if_not_exist=True)
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample['nsentences']

    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning("BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization")
            else:
                logger.warning("If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization")
                # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
                print(
                    "Generate {} with beam={}: {}".format(
                        cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
                    ),
                    file=output_file,
                )

    return scorer


@dataclass
class LMFusionConfig(FairseqDataclass):
    primary_lm_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to lm checkpoint for primary lm fusion"},
    )
    primary_lm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for lm probs for primary lm fusion"},
    )
    auxiliary_lm_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to lm checkpoint for auxiliary lm fusion"},
    )
    auxiliary_lm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for lm probs for auxiliary lm fusion"},
    )
    encoder_avg_outs: Optional[str] = field(
        default=None,
        metadata={"help": "path to encoder avgs"},
    )
    primary_ilm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for ilm probs for primary ilm removal"},
    )
    auxiliary_ilm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for ilm probs for auxiliary ilm removal"},
    )


def cli_main():
    parser = options.get_generation_parser()
    lm_group = parser.add_argument_group("LM fusion")
    gen_parser_from_dataclass(lm_group, LMFusionConfig())
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
