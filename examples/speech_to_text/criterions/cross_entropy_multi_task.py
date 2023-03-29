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


@register_criterion("cross_entropy_multi_task")
class CrossEntropyMultitask(LabelSmoothedCrossEntropyCriterion):
    """
    Loss designed for multitask models with an auxiliary classifier.
    It computes the cross-entropy loss on the classifier output and
    label-smoothed cross entropy on the base model output.
    """
    def __init__(self, args, task):
        super().__init__(
            task, args.sentence_avg, args.label_smoothing, args.ignore_prefix_size)
        self.auxiliary_loss_weight = args.auxiliary_loss_weight
        if args.auxiliary_loss_class_weights is not None:
            self.auxiliary_loss_class_weights = torch.FloatTensor(args.auxiliary_loss_class_weights)
        else:
            self.auxiliary_loss_class_weights = None

    @classmethod
    def build_criterion(cls, args, task):
        return cls(args, task)

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--auxiliary-loss-weight', default=1.0, type=float, metavar='W',
                            help='The weight to apply to the auxiliary loss function when summing losses')
        parser.add_argument('--auxiliary-loss-class-weights', default=None, type=float, nargs="+", metavar='Ws',
                            help='Individual class weights for balancing uneven classes')

    def forward(self, model, sample, reduce=True, log_probs=True):
        net_output = model(**sample['net_input'])
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        main_loss, nll_loss = self.compute_loss(model, net_output[0], sample)
        auxiliary_probs = model.auxiliary_decoder.get_normalized_probs(net_output[1], log_probs=True)
        if self.auxiliary_loss_class_weights is not None:
            class_weights = self.auxiliary_loss_class_weights.to(sample["auxiliary_target"].device)
        else:
            class_weights = None
        auxiliary_loss = F.nll_loss(
            auxiliary_probs,
            sample["auxiliary_target"].view(-1),
            weight=class_weights,
            reduction='sum' if reduce else 'none',)

        n_correct = torch.sum(
            auxiliary_probs.argmax(1).eq(sample["auxiliary_target"].view(-1)))

        loss = main_loss + self.auxiliary_loss_weight * auxiliary_loss
        logging_output = {
            'loss': loss.data,
            'main_loss': main_loss.data,
            'nll_loss': nll_loss.data,
            'auxiliary_loss': auxiliary_loss.data,
            'ntokens': sample['ntokens'],
            'n_correct': n_correct,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed():
        return True

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        main_loss_sum = sum(log.get('main_loss', 0) for log in logging_outputs)
        auxiliary_loss_sum = sum(log.get('auxiliary_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        n_correct = sum(log.get('n_correct', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('main_loss', main_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('auxiliary_loss', auxiliary_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('accuracy', n_correct / nsentences, nsentences, round=4)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
