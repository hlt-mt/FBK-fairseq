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
import torch
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


def label_smoothed_nll_loss(
        lprobs,
        target,
        is_tagged,
        tag_weight,
        notag_weight,
        epsilon,
        ignore_index=None,
        reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        is_tagged = is_tagged.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    weights = torch.zeros(is_tagged.shape, dtype=nll_loss.dtype).to(nll_loss.device)
    weights[is_tagged] = tag_weight
    weights[~is_tagged] = notag_weight
    nll_loss = nll_loss * weights
    smooth_loss = smooth_loss * weights
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
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


@register_criterion("weighted_label_smoothed_cross_entropy")
class WeightedLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    """
    The criterion allows the user to define a different weight for data in tags
    and data outside tags. E.g. it can be used to give higher weight to named entities
    tagged in the training set with respect to the rest of the text.
    """
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        tag_weight,
        notag_weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.tag_weight = tag_weight
        self.notag_weight = notag_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--tag-weight', default=1., type=float, metavar='D',
                            help='weight to give to tagged items')
        parser.add_argument('--notag-weight', default=1., type=float, metavar='D',
                            help='weight to give to non-tagged items')
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            (sample["target_tags"] > 0).view(-1),
            self.tag_weight,
            self.notag_weight,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
