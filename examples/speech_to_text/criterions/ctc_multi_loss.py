import math
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from fairseq import utils, metrics
from fairseq.criterions import register_criterion, LegacyFairseqCriterion, FairseqCriterion
from fairseq.criterions.ctc import CtcCriterion


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

    # To support dualdecoder models with CTC, we include the following methods
    @property
    def auxiliary_decoder(self):
        return self.model.auxiliary_decoder

    def get_auxiliary_target(self, sample, auxiliary_output):
        return self.model.get_auxiliary_target(sample, auxiliary_output)

    def get_auxiliary_token_lens(self, sample):
        return self.model.get_auxiliary_token_lens(sample)

    # To support joint CTC decoding
    def get_auxiliary_input_lens(self, sample, net_output):
        return self.model.get_auxiliary_input_lens(sample, net_output)

    @property
    def auxiliary_dict(self):
        return self.model.auxiliary_dict


class BaseCTCLoss(CtcCriterion):
    def __init__(self, args, task):
        super(FairseqCriterion, self).__init__(task)
        self.args = args
        self.blank_idx = task.source_dictionary.index("<ctc_blank>")
        self.pad_idx = task.source_dictionary.pad()
        self.eos_idx = task.source_dictionary.eos()

        self.post_process = self.args.ctc_post_process

        if self.args.wer_args is not None:
            (
                self.args.wer_kenlm_model,
                self.args.wer_lexicon,
                self.args.wer_lm_weight,
                self.args.wer_word_score,
            ) = eval(self.args.wer_args)

        if self.args.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = self.args.wer_kenlm_model
            dec_args.lexicon = self.args.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = self.args.wer_lm_weight
            dec_args.word_score = self.args.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = self.args.zero_infinity
        self.sentence_avg = self.args.sentence_avg


@register_criterion("ctc_multi_loss")
class CTCMultiLoss(LegacyFairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        assert task.source_dictionary is not None
        self.ctc_criterion = BaseCTCLoss(args, task)
        self.real_criterion = CTCMultiLoss.build_real_criterion(args, task)
        self.__class__.real_criterion_class = self.real_criterion.__class__
        self.ctc_weight = args.ctc_weight

    @staticmethod
    def build_real_criterion(args, task):
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion
        assert saved_criterion != args.underlying_criterion
        underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        return underlying_criterion

    @staticmethod
    def add_args(parser):
        parser.add_argument('--ctc-encoder-layer', default=6, type=int, metavar='LAYER_NUM',
                            help='The encoder layer whose feature are used to compute the CTC loss')
        parser.add_argument('--ctc-weight', default=1.0, type=float, metavar='W',
                            help='The relative weight to assign to the CTC loss')
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True,
                            help='underlying criterion to use for the model output loss')
        parser.add_argument('--zero-infinity', default=True, type=bool, metavar='ZERO_INF',
                            help='zero inf loss when source length <= target length')
        parser.add_argument('--ctc-post-process', default='letter', metavar='POST_PROC',
                            help='how to post process predictions into words. can be letter, wordpiece, BPE symbols, etc. \
            See fairseq.data.data_utils.post_process() for full list of options')
        parser.add_argument('--wer-kenlm-model', default=None, metavar='WER_KENLM',
                            help='if this is provided, use kenlm to compute wer (along with other wer_* args)')
        parser.add_argument('--wer-lexicon', default=None, metavar='WER_LEX',
                            help='lexicon to use with wer_kenlm_model')
        parser.add_argument('--wer-lm-weight', default=2.0, metavar='WER_LM_W',
                            help='lm weight to use with wer_kenlm_model')
        parser.add_argument('--wer-word-score', default=1.0, metavar='WER_WORD_SCORE',
                            help='lm word score to use with wer_kenlm_model')
        parser.add_argument('--wer-args', default=None, metavar='WER_ARGS',
                            help='DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)')

    def forward(self, model, sample, reduce=True):
        decoder_out, encoder_out = model(**sample["net_input"])
        encoder_fake_model = FakeEncoderModel(model.encoder, encoder_out, sample["transcript"])
        decoder_fake_model = FakeDecoderModel(model, decoder_out, sample["target"])
        encoder_sample = {
            "net_input": {
                "src_lengths": encoder_out["ctc_lengths"]
            },
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

        logging_output = {k: v for k, v in real_logging_output.items() if k != "loss"}
        logging_output["ctc_loss"] = utils.item(ctc_logging_output['loss'])
        logging_output["real_loss"] = utils.item(real_logging_output['loss'])
        logging_output["loss"] = utils.item(loss.data) if reduce else loss.data
        return loss, ctc_sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed():
        return True

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        cls.real_criterion_class.reduce_metrics(logging_outputs)
        real_loss_sum = utils.item(sum(log.get('real_loss', 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get('ctc_loss', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('real_loss', real_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ctc_loss', ctc_loss_sum / sample_size / math.log(2), sample_size, round=3)
