# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace

import torch

from fairseq.data import Dictionary, encoders, data_utils
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
)
from fairseq.sequence_generator import EnsembleModelWithAlignment, SequenceGenerator
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)

@register_task("speech_to_text")
class SpeechToTextTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--extract-attn-from-layer",
            metavar="W",
            default=4,  # the 5th layer, following Garg et al. 2019
            type=int,
            help="layer used to get the encoder-decoder attention scores",
        )

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))
        self.extract_attn_from_layer = getattr(args, 'extract_attn_from_layer', None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids
        if seq_gen_cls is None and getattr(args, "print_alignment", False):
            seq_gen_cls = SequenceGeneratorSpeechWithAlignment
            extra_gen_cls_kwargs['print_alignment'] = args.print_alignment
            extra_gen_cls_kwargs['extract_attn_from_layer'] = self.extract_attn_from_layer
        return super().build_generator(
            models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    @classmethod
    def build_dataset_for_inference(cls, audio_paths, n_frames):
        return SpeechToTextDataset("interactive", False, {}, audio_paths, n_frames)


class SequenceGeneratorSpeechWithAlignment(SequenceGenerator):
    def __init__(
            self,
            models,
            tgt_dict,
            left_pad_target=False,
            print_alignment="soft",
            extract_attn_from_layer=None,
            **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target
        self.extract_attn_from_layer = extract_attn_from_layer
        assert print_alignment == "soft", "Only supporting soft for speech source"

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        kwargs["extract_attn_from_layer"] = self.extract_attn_from_layer
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, tgt_tokens = self._prepare_batch_for_alignment(sample, finalized)

        attn = [
            finalized[i // beam_size][i % beam_size]["attention"].transpose(1, 0)
            for i in range(bsz * beam_size)
        ]

        if tgt_tokens.device != "cpu":
            tgt_tokens = tgt_tokens.to("cpu")
            attn = [i.to("cpu") for i in attn]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            finalized_i = finalized[i // beam_size][i % beam_size]
            alignment = self.extract_alignment(
                attn[i], src_lengths[i], tgt_tokens[i], self.pad, finalized_i["ctc_batch_predicted"])
            finalized_i["alignment"] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :, :]
            .expand(-1, self.beam_size, -1, src_tokens.shape[-1])
            .contiguous()
            .view(bsz * self.beam_size, -1, src_tokens.shape[-1])
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
            .expand(-1, self.beam_size)
            .contiguous()
            .view(bsz * self.beam_size)
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, tgt_tokens

    def extract_alignment(self, attn, src_len, tgt_sent, pad, ctc_batch_predicted):
        tgt_valid = (tgt_sent != pad).nonzero(as_tuple=False)
        if len(ctc_batch_predicted) > 0:
            src_len = len(ctc_batch_predicted)
        else:
            src_len = torch.ceil(src_len / 4).item()
        src_valid = torch.arange(attn.shape[1]).to(attn.device)
        src_valid = src_valid < src_len
        alignment = []
        if len(tgt_valid) != 0 and len(src_valid) != 0:
            attn_valid = attn[tgt_valid, src_valid]
            alignment = [
                ["{:.6f}".format(p) for p in src_probs.tolist()] for src_probs in attn_valid]
        return alignment
