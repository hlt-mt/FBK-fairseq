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
import math

import torch

from examples.speech_to_text.criterions.cross_entropy_dualdecoder import CrossEntropyDualDecoder
from fairseq import metrics
from fairseq.criterions import register_criterion


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


@register_criterion("cross_entropy_dualdecoder_with_tags")
class CrossEntropyDualDecoderWithTags(CrossEntropyDualDecoder):
    """
    This criterion computes a multitask loss that comprises 4 label smoothed cross entropy (LSCE):
     - LSCE between the auxiliary decoder outputs and the transcripts;
     - LSCE between the main decoder outputs and the translations;
     - LSCE between the tag type of each transcript token and the "tags" output of the auxiliary decoder
     - LSCE between the tag type of each translation token and the "tags" output of the main decoder.
     The 4 LSCE are summed to obtain the total loss, weighted according to the values set for
     the corresponding hyper-parameters (all set to 0.5 by default).
    """
    def __init__(self, args, task):
        super().__init__(args, task)
        self.auxiliary_tags_loss_weight = args.auxiliary_tags_loss_weight
        self.primary_tags_loss_weight = args.primary_tags_loss_weight

    @staticmethod
    def add_args(parser):
        CrossEntropyDualDecoder.add_args(parser)
        parser.add_argument('--primary-tags-loss-weight', default=0.5, type=float, metavar='W',
                            help='The weight to apply to the loss function of the tags from the primary decoder '
                                 'when summing losses')
        parser.add_argument('--auxiliary-tags-loss-weight', default=0.5, type=float, metavar='W',
                            help='The weight to apply to the loss function of the tags from the auxiliary decoder '
                                 'when summing losses')

    def compute_loss(self, net_output, model, sample, reduce=True, log_probs=True):
        txt_loss, sample_size, logging_output = super(CrossEntropyDualDecoderWithTags, self).compute_loss(
            net_output, model, sample, reduce=reduce, log_probs=log_probs)
        primary_tags_lprobs = torch.log_softmax(net_output[0][1]["tags"], dim=-1)
        auxiliary_tags_lprobs = torch.log_softmax(net_output[1][1]["tags"], dim=-1)
        primary_tags_loss, primary_tags_nll_loss = masked_label_smoothed_ce(
            primary_tags_lprobs.view(-1, primary_tags_lprobs.size(-1)),
            sample["target_tags"].view(-1, 1),
            self.eps,
            pad_mask=sample["target"].view(-1, 1).eq(self.padding_idx),
            reduce=reduce
        )
        auxiliary_tags_loss, auxiliary_tags_nll_loss = masked_label_smoothed_ce(
            auxiliary_tags_lprobs.view(-1, auxiliary_tags_lprobs.size(-1)),
            sample["transcript_tags"].view(-1, 1),
            self.eps,
            pad_mask=sample["transcript"].view(-1, 1).eq(self.padding_idx),
            reduce=reduce
        )
        loss = txt_loss + self.primary_tags_loss_weight * primary_tags_loss + \
            self.auxiliary_tags_loss_weight * auxiliary_tags_loss
        logging_output["loss"] = loss.data
        logging_output["auxiliary_tags_loss"] = auxiliary_tags_loss.data
        logging_output["primary_tags_loss"] = primary_tags_loss.data
        logging_output["auxiliary_tags_nll_loss"] = auxiliary_tags_nll_loss.data
        logging_output["primary_tags_nll_loss"] = primary_tags_nll_loss.data
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        CrossEntropyDualDecoder.reduce_metrics(logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        auxiliary_ntokens = sum(log.get('auxiliary_ntokens', 0) for log in logging_outputs)
        primary_tags_loss = sum(log.get('primary_tags_loss', 0) for log in logging_outputs)
        auxiliary_tags_loss = sum(log.get('auxiliary_tags_loss', 0) for log in logging_outputs)
        primary_tags_nll_loss = sum(log.get('primary_tags_nll_loss', 0) for log in logging_outputs)
        auxiliary_tags_nll_loss = sum(log.get('auxiliary_tags_nll_loss', 0) for log in logging_outputs)

        metrics.log_scalar(
            'primary_tags_loss',
            primary_tags_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar(
            'auxiliary_tags_loss',
            auxiliary_tags_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar(
            'primary_tags_nll_loss',
            primary_tags_nll_loss / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar(
            'auxiliary_tags_nll_loss',
            auxiliary_tags_nll_loss / auxiliary_ntokens / math.log(2), auxiliary_ntokens, round=3)
