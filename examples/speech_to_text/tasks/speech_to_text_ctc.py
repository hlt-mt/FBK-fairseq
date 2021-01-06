# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os.path as op

from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc, S2TDataConfigSrc, \
SpeechToTextDatasetCreatorWithSrc
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask

logger = logging.getLogger(__name__)


@register_task("speech_to_text_ctc")
class SpeechToTextCtcTask(SpeechToTextTask):
    """
    Task for training speech recognition model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechToTextTask.add_args(parser)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.data_cfg = S2TDataConfigSrc(op.join(args.data, args.config_yaml))

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        data_cfg = S2TDataConfigSrc(op.join(args.data, args.config_yaml))

        # target dict
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )
        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        # Source dict
        source_dict_path = op.join(args.data, data_cfg.vocab_filename_src)
        src_dict = Dictionary.load(source_dict_path)
        if args.criterion == "ctc_multi_loss":
            src_dict.add_symbol("<ctc_blank>")
        logger.info(
            f"source dictionary size ({data_cfg.vocab_filename_src}): " f"{len(src_dict):,}"
        )
        return cls(args, tgt_dict, src_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreatorWithSrc.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    @property
    def source_dictionary(self):
        return self.src_dict
