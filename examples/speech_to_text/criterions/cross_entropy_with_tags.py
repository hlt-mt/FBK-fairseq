# Copyright 2022 FBK

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

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


def masked_label_smoothed_ce(lprobs, target, epsilon, pad_mask=None, reduce=True):
    """
    A label-smoothing cross entropy implementation that is aware of the padding mask.
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if pad_mask is not None:
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("cross_entropy_with_tags")
class CrossEntropyDualDecoderWithTags(LabelSmoothedCrossEntropyCriterion):
    """
    This criterion computes a multitask loss that comprises 2 label smoothed cross entropy (LSCE):
     - LSCE between the main decoder outputs and the translations;
     - LSCE between the tag type of each translation token and the "tags" output of the main decoder.
     The additional tag LSCE is summed to the main loss, weighted according to the value set for
     the corresponding hyper-parameter (set to 0.5 by default).
    """
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            tags_loss_weight,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.tags_loss_weight = tags_loss_weight

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--tags-loss-weight', default=0.5, type=float, metavar='W',
                            help='The weight to apply to the loss function of the tags')

    def tags_loss(self, net_output, sample, reduce=True):
        tags_lprobs = torch.log_softmax(net_output[1]["tags"], dim=-1)
        tags_loss, tags_nll_loss = masked_label_smoothed_ce(
            tags_lprobs.view(-1, tags_lprobs.size(-1)),
            sample["target_tags"].view(-1, 1),
            self.eps,
            pad_mask=sample["target"].view(-1, 1).eq(self.padding_idx),
            reduce=reduce
        )
        return tags_loss, tags_nll_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce=reduce)
        tags_loss, tags_nll_loss = self.tags_loss(net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        loss = loss + self.tags_loss_weight * tags_loss
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "tags_loss": tags_loss.data,
            "primary_tags_nll_loss": tags_nll_loss.data,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        tags_loss = sum(log.get('tags_loss', 0) for log in logging_outputs)
        tags_nll_loss = sum(log.get('tags_nll_loss', 0) for log in logging_outputs)

        metrics.log_scalar(
            'tags_loss',
            tags_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar(
            'tags_nll_loss',
            tags_nll_loss / ntokens / math.log(2), ntokens, round=3)
