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
import ast
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter


class AccuracyScorer:
    """
    Computes total and class-based accuracy.
    """
    def __init__(self):
        self.class_stats = {}

    def add(self, ref, pred):
        if ref not in self.class_stats:
            self.class_stats[ref] = {"num_correct": 0, "total": 0}
        if ref == pred:
            self.class_stats[ref]["num_correct"] += 1
        self.class_stats[ref]["total"] += 1

    def total(self):
        return sum(s["total"] for s in self.class_stats.values())

    def total_correct(self):
        return sum(s["num_correct"] for s in self.class_stats.values())

    def score(self):
        total = self.total()
        if total == 0:
            return 0.0
        return float(self.total_correct()) / total

    def class_accuracy(self, cls):
        if self.class_stats[cls]["total"] == 0:
            return 0.0
        return float(self.class_stats[cls]["num_correct"]) / self.class_stats[cls]["total"]

    def result_string(self, classes=None):
        class_strings = []
        keys = sorted(self.class_stats.keys())
        for class_id in keys:
            if classes is not None:
                class_name = classes[class_id]
            else:
                class_name = str(class_id)
            class_strings.append(f"'{class_name}': {self.class_accuracy(class_id):.2f}")
        return f"Accuracy: {self.score():.2f}. Class accuracy: {{{', '.join(class_strings)}}}"


def generate_probs(models, sample):
    """
    Returns the classification probabilities for the passed samples.
    """
    encoder_input = {
        k: v
        for k, v in sample['net_input'].items()
        if k != "prev_output_tokens"
    }
    encoder_outs = [model.encoder.forward(**encoder_input) for model in models]
    probs = []
    for model, enc_out in zip(models, encoder_outs):
        probs.append(model.auxiliary_decoder.get_normalized_probs(
            model.auxiliary_decoder.forward(enc_out), log_probs=False))
    if len(probs) == 1:
        return probs[0]
    avg_probs = torch.mean(torch.stack(probs, dim=0), dim=0)
    return avg_probs


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"

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
    This function works similarly to the generate.py, but instead of generating the output
    with the given model, it computes the probability of the classifier on top of the
    encoder output. It works only for multitask models with a classifier as *auxiliary_decoder*.
    """
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

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

    # loading the dataset should happen after the checkpoint has been loaded,
    # so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Optimize ensemble for generation
    for model in chain(models):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
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

    gen_timer = StopwatchMeter()
    scorer = AccuracyScorer()
    num_sentences = 0
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        with torch.no_grad():
            gen_timer.start()
            hypos = generate_probs(models, sample).cpu()
            gen_timer.stop(1)
            num_sentences += sample['nsentences']
            for i, sample_id in enumerate(sample["id"].tolist()):
                reference = sample['auxiliary_target'][i].item()
                predicted = hypos[i].argmax()
                if not cfg.common_eval.quiet:
                    print('S-{}\t{}'.format(sample_id, reference), file=output_file)

                print('H-{}\t{}\t{}'.format(sample_id, predicted, hypos[i]), file=output_file)
                scorer.add(reference, predicted)
    logger.info('Predicted {} sentences in {:.1f}s ({:.2f} sentences/s)'.format(
        num_sentences, gen_timer.sum, num_sentences / gen_timer.sum))
    classes = getattr(task.data_cfg, 'aux_classes')
    print(f"Generate {cfg.dataset.gen_subset}: {scorer.result_string(classes=classes)}")


def cli_main():
    """
    Returns the classification probability of the classifier on top of
    the encoder outputs for the multitask classifier.

    Uses the same options of the generate.py
    """
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
