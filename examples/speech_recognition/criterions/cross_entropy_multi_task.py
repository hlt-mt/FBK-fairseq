import math

import torch
import torch.nn.functional as F


from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion("cross_entropy_multi_task")
class CrossEntropyMultitask(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)
        self.eps = args.label_smoothing
        self.auxiliary_loss_weight = args.auxiliary_loss_weight
        self.sentence_avg = args.sentence_avg
        if args.auxiliary_loss_class_weights is not None:
            self.auxiliary_loss_class_weights = torch.FloatTensor(args.auxiliary_loss_class_weights)
        else:
            self.auxiliary_loss_class_weights = None

    @classmethod
    def build_criterion(cls, args, task):
        return cls(args, task)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--auxiliary-loss-weight', default=1.0, type=float, metavar='W',
                            help='The weight to apply to the auxiliary loss function when summing losses')
        parser.add_argument('--auxiliary-loss-class-weights', default=None, type=float, nargs="+", metavar='Ws',
                            help='Individual class weights for balancing uneven classes')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True, log_probs=True):
        net_output = model(**sample['net_input'])
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        lprobs = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output[0]).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
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
        loss = loss + self.auxiliary_loss_weight * auxiliary_loss
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'auxiliary_loss': auxiliary_loss.data,
            'ntokens': sample['ntokens'],
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
        auxiliary_loss_sum = sum(log.get('auxiliary_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('auxiliary_loss', auxiliary_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
