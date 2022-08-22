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
import logging
import os.path as op

from examples.speech_to_text.data.speech_to_text_dataset_tagged import S2TDataConfigTagged, \
    SpeechToTextDatasetTaggedCreator
from examples.speech_to_text.inference.sequence_generator_tagged import SequenceGeneratorWithTags
from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from examples.speech_to_text.tasks.speech_translation_dualdecoding import SpeechTranslationDualDecodingTask
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task("speech_to_text_ctc_tagged")
class SpeechToTextTaggedTask(SpeechToTextCtcTask):
    """
    Task for training/inference with seq2seq models on speech data tagged with named entities.
    """

    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        self.data_cfg = S2TDataConfigTagged(op.join(args.data, args.config_yaml))

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        bpe_tokenizer_src = self.build_bpe_src(self.args)
        self.datasets[split] = SpeechToTextDatasetTaggedCreator.from_tsv(
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

    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
    ):
        return super().build_generator(
            models,
            args,
            seq_gen_cls=SequenceGeneratorWithTags,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs)


@register_task("speech_to_text_tagged_dual")
class SpeechToTextTaggedDualdecodingTask(SpeechTranslationDualDecodingTask):
    """
    Task for training/inference with triangle models on speech data tagged with named entities.
    """

    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        self.data_cfg = S2TDataConfigTagged(op.join(args.data, args.config_yaml))

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        bpe_tokenizer_src = self.build_bpe_src(self.args)
        self.datasets[split] = SpeechToTextDatasetTaggedCreator.from_tsv(
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
