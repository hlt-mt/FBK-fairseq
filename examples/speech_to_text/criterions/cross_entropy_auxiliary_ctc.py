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
import math

import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion("joint_cross_entropy_ctc")
class JointCrossEntropyCtcLoss(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            auxiliary_loss_weight=0.5,
            primary_loss_weight=0.5):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.primary_loss_weight = primary_loss_weight
        self.eos_idx = task.target_dictionary.eos()

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--primary-loss-weight', default=0.5, type=float, metavar='W',
                            help='The weight to apply to the primary loss function when summing losses')
        parser.add_argument('--auxiliary-loss-weight', default=0.5, type=float, metavar='W',
                            help='The weight to apply to the auxiliary loss function when summing losses')

    def compute_auxiliary_loss(self, model, net_output, sample):
        lprobs = model.auxiliary_decoder.get_normalized_probs(
            net_output, log_probs=True).contiguous()  # (T, B, C) from the encoder

        target = model.get_auxiliary_target(sample, net_output)
        input_lengths = model.get_auxiliary_input_lens(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                target = target[self.ignore_prefix_size:, :].contiguous()

        pad_mask = (target != self.padding_idx) & (target != self.eos_idx)
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.task.target_dictionary.index("<ctc_blank>"),
                reduction="sum",
                zero_infinity=True)
        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        primary_loss, nll_loss = self.compute_loss(model, net_output[0], sample, reduce=reduce)
        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        auxiliary_loss = self.compute_auxiliary_loss(model, net_output[1], sample)

        loss = self.primary_loss_weight * primary_loss + self.auxiliary_loss_weight * auxiliary_loss
        logging_output = {
            "loss": loss.data,
            "primary_loss": primary_loss.data,
            "auxiliary_loss": auxiliary_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        primary_loss_sum = sum(log.get("primary_loss", 0) for log in logging_outputs)
        auxiliary_loss_sum = sum(log.get("auxiliary_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar(
            'primary_loss', primary_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar(
            'auxiliary_loss', auxiliary_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
