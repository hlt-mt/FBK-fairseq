import math

import torch
import torch.nn.functional as F
from torch import nn

from fairseq import utils, metrics
from fairseq.criterions import register_criterion, LegacyFairseqCriterion
from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
from omegaconf import II


class FakeEncoderModel(nn.Module):
    def __init__(self, encoder, net_out, target):
        super().__init__()
        self.net_out = net_out
        self.target = target
        if hasattr(encoder, "output_batch_first"):
            self.output_batch_first = encoder.output_batch_first

    def forward(self, **unused):
        return self.net_out

    def get_targets(self, *unused):
        return self.target

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["ctc_out"]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                probs = F.log_softmax(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            if hasattr(self, "output_batch_first"):
                probs.batch_first = self.output_batch_first
            return probs
        raise NotImplementedError


class FakeDecoderModel(nn.Module):
    def __init__(self, model, net_out, target):
        super().__init__()
        self.model = model
        self.net_out = net_out
        self.target = target

    def forward(self, **unused):
        return self.net_out

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.model.get_normalized_probs(net_output, log_probs, sample=sample)

    def get_targets(self, *unused):
        return self.target

    @property
    def decoder(self):
        return self.model.decoder


class BaseCTCLoss(CtcCriterion):
    def __init__(self, args, task):
        cfg = CtcCriterionConfig()
        super().__init__(cfg, task)
        self.args = args
        self.zero_infinity = self.args.zero_infinity
        self.sentence_avg = self.args.sentence_avg
        self.post_process = self.args.post_process
        self.wer_kenlm_model = self.args.wer_kenlm_model
        self.wer_lexicon = self.args.wer_lexicon
        self.wer_lm_weight = self.args.wer_lm_weight
        self.wer_word_score = self.args.wer_word_score
        self.wer_args = self.args.wer_args
        self.blank_idx = task.source_dictionary.index("<ctc_blank>")
        self.pad_idx = task.source_dictionary.pad()
        self.eos_idx = task.source_dictionary.eos()

@register_criterion("ctc_multi_loss")
class CTCMultiLoss(LegacyFairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        assert task.source_dictionary is not None
        self.ctc_criterion = BaseCTCLoss(args, task)
        self.real_criterion = CTCMultiLoss.build_real_criterion(args, task)
        self.ctc_weight = args.ctc_weight

    @staticmethod
    def build_real_criterion(args, task):
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion
        assert saved_criterion != args.underlying_criterion
        #underlying_criterion = task.build_criterion(task)
        underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        return underlying_criterion

    @staticmethod
    def add_args(parser):
        CtcCriterion.add_args(parser)
        parser.add_argument('--ctc-encoder-layer', default=6, type=int, metavar='LAYER_NUM',
                            help='The encoder layer whose feature are used to compute the CTC loss')
        parser.add_argument('--ctc-weight', default=1.0, type=float, metavar='W',
                            help='The relative weight to assign to the CTC loss')
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True,
                            help='underlying criterion to use for the model output loss')
        parser.add_argument('--zero_infinity', default=True, type=bool, metavar='ZERO_INF',
                            help='zero inf loss when source length <= target length')
        parser.add_argument('--sentence_avg', default=II("optimization.sentence_avg"), metavar="SEN_AVG",
                            )
        parser.add_argument('--post_process', default='letter', metavar='POST_PROC',
                            help='how to post process predictions into words. can be letter, wordpiece, BPE symbols, etc. \
            See fairseq.data.data_utils.post_process() for full list of options')
        parser.add_argument('--wer_kenlm_model', default=None, metavar='WER_KENLM',
                            help='if this is provided, use kenlm to compute wer (along with other wer_* args)')
        parser.add_argument('--wer_lexicon', default=None, metavar='WER_LEX',
                            help='lexicon to use with wer_kenlm_model')
        parser.add_argument('--wer_lm_weight', default=2.0, metavar='WER_LM_W',
                            help='lm weight to use with wer_kenlm_model')
        parser.add_argument('--wer_word_score', default=1.0, metavar='WER_WORD_SCORE',
                            help='lm word score to use with wer_kenlm_model')
        parser.add_argument('--wer_args', default=None, metavar='WER_ARGS',
                            help='DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)')

    def forward(self, model, sample, reduce=True):
        decoder_out, encoder_out = model(**sample["net_input"])
        encoder_fake_model = FakeEncoderModel(model.encoder, encoder_out, sample["transcript"])
        decoder_fake_model = FakeDecoderModel(model, decoder_out, sample["target"])
        fake_encoder_net_input = {
            "src_lengths": encoder_out["ctc_lengths"]
        }
        encoder_sample = {
            "net_input": fake_encoder_net_input,
            "target": sample["transcript"],
            "target_lengths": sample["transcript_lengths"]-1,
            "ntokens": sum(sample["transcript_lengths"]).item(),
            "id": sample["id"]
        }
        ctc_loss, ctc_sample_size, ctc_logging_output = self.ctc_criterion(
            encoder_fake_model, encoder_sample, reduce=reduce)
        real_loss, _, real_logging_output = self.real_criterion(
            decoder_fake_model, sample, reduce=reduce)
        loss = self.ctc_weight * ctc_loss + real_loss

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ctc_loss": ctc_logging_output['loss'],
            "ntokens": real_logging_output['ntokens'],
            "nsentences": real_logging_output['nsentences'],
            "sample_size": real_logging_output['sample_size'],
        }
        if 'nll_loss' in real_logging_output:
            logging_output['nll_loss'] = real_logging_output['nll_loss']
        return loss, ctc_sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed():
        return True

    @staticmethod
    def reduce_metrics(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get('ctc_loss', 0) for log in logging_outputs))
        if logging_outputs and 'nll_loss' in logging_outputs[0]:
            nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        else:
            nll_loss_sum = loss_sum - ctc_loss_sum  # NLL computed on the real loss, not on the auxiliary CTC
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('ctc_loss', ctc_loss_sum / sample_size / math.log(2), sample_size, round=3)