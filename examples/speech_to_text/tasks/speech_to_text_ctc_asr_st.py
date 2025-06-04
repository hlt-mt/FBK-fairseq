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

import logging
import math
import random

import torch
import warnings

from examples.speech_to_text.data.speech_to_text_dataset_asr_st import SpeechToTextDatasetCreatorASRST
from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from fairseq.tasks import register_task
from fairseq import utils, metrics

logger = logging.getLogger(__name__)


@register_task("speech_to_text_ctc_asr_st")
class SpeechToTextCtcASRSTTask(SpeechToTextCtcTask):
    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        self.p_sampling_asr = args.p_sampling_asr

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechToTextCtcTask.add_args(parser)
        parser.add_argument(
            "--p-sampling-asr", type=float, default=0.5,
            help="Sampling rate (from 0 to 1) for the ASR target during training. Default to 0.5.")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        bpe_tokenizer_src = self.build_bpe_src(self.args)
        self.datasets[split] = SpeechToTextDatasetCreatorASRST.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            bpe_tokenizer_src,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        """
        Do forward and backward for both ST and ASR consecutively, and return the composite loss as
         computed by *criterion* for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the composite ASR and ST loss
                - the sample size, which is used as the denominator for the
                  gradient, resulting from both ASR and ST sample size
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        asr_step = False

        # Generate a probability p from a uniform distribution between 0 and 1
        p = random.uniform(0, 1)

        # If p is lower than p_sampling_asr, ASR training is performed, otherwise ST
        if p < self.p_sampling_asr:
            asr_step = True
            sample["net_input"]["prev_output_tokens"] = sample["net_input"]["prev_transcript_tokens"]
            sample["target"] = sample["prepended_transcript"]
            sample["target_lengths"] = sample["prepended_transcript_lengths"]
            sample["ntokens"] = sample["ntokens_prepended_transcript"]

        # Forward and backward passes
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        if asr_step:
            logging_output = {k + "_asr": v for k, v in logging_output.items()}
        else:
            logging_output = {k + "_st": v for k, v in logging_output.items()}

        logging_output["loss"] = utils.item(loss.data)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        """
        Do validation for both ST and ASR consecutively, and return the composite loss as computed
        by *criterion* for the given *model* and *sample*.
        """
        model.eval()
        # ST validation
        with torch.no_grad():
            loss_st, sample_size_st, logging_output_st = criterion(model, sample)
        # ASR validation
        sample_asr = sample
        sample_asr["net_input"]["prev_output_tokens"] = sample["net_input"]["prev_transcript_tokens"]
        sample_asr["target"] = sample["prepended_transcript"]
        sample_asr["target_lengths"] = sample["prepended_transcript_lengths"]
        sample_asr["ntokens"] = sample["ntokens_prepended_transcript"]
        with torch.no_grad():
            loss_asr, sample_size_asr, logging_output_asr = criterion(model, sample_asr)

        # Composite loss, sample size, and outputs
        loss = loss_asr + loss_st
        sample_size = sample_size_st + sample_size_asr

        logging_output_st = {k + "_st": v for k, v in logging_output_st.items()}
        logging_output_asr = {k + "_asr": v for k, v in logging_output_asr.items()}

        logging_output = logging_output_st
        logging_output.update(logging_output_asr)

        logging_output["loss"] = utils.item(loss.data)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size_asr = sum(log.get("sample_size_asr", 0) for log in logging_outputs)
        sample_size_st = sum(log.get("sample_size_st", 0) for log in logging_outputs)
        sample_size = sample_size_asr + sample_size_st

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size_asr > 0:
            self.reduce_label_smoothing_and_ctc_metrics(logging_outputs, "_asr")
        if sample_size_st > 0:
            self.reduce_label_smoothing_and_ctc_metrics(logging_outputs, "_st")

    @classmethod
    def reduce_label_smoothing_and_ctc_metrics(cls, logging_outputs, suffix) -> None:
        sample_size = sum(log.get('sample_size' + suffix, 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss' + suffix, 0) for log in logging_outputs)
        metrics.log_scalar(
            'loss' + suffix, loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0:
            if 'nll_loss' + suffix in logging_outputs[0]:
                nll_loss_sum = sum(log.get("nll_loss" + suffix, 0) for log in logging_outputs)
                ntokens = sum(log.get("ntokens" + suffix, 0) for log in logging_outputs)
                metrics.log_scalar(
                    "nll_loss" + suffix, nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
                metrics.log_derived(
                    "ppl" + suffix, lambda meters: utils.get_perplexity(meters["nll_loss" + suffix].avg))
            # these are optional as they are used in the ctc multi loss and multi-task training
            for prefix in ['real_loss', 'ctc_loss', 'primary_loss', 'auxiliary_loss']:
                if prefix + suffix in logging_outputs[0]:
                    prefix_logs_sum = utils.item(sum(
                        log.get(prefix + suffix, 0) for log in logging_outputs))
                    metrics.log_scalar(
                        prefix + suffix,
                        prefix_logs_sum / sample_size / math.log(2),
                        sample_size,
                        round=3)
