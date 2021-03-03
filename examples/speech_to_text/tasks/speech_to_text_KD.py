# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os.path as op

from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask

from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc

from examples.speech_to_text.data.speech_to_text_dataset_KD import SpeechToTextDatasetCreatorKD
from fairseq.tasks import register_task


logger = logging.getLogger(__name__)


@register_task("speech_to_text_kd")
class SpeechToTextTaskKD(SpeechToTextCtcTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechToTextCtcTask.add_args(parser)
        parser.add_argument('--distill-topk', default=None, type=int, required=True, metavar='K',
                            help='value of k')

    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        self.src_dict = src_dict
        self.data_cfg = S2TDataConfigSrc(op.join(args.data, args.config_yaml))
        self.src_speech = True

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        bpe_tokenizer_src = self.build_bpe_src(self.args)
        self.datasets[split] = SpeechToTextDatasetCreatorKD.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            bpe_tokenizer_src,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

