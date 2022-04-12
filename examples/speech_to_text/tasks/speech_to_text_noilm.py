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
from examples.speech_to_text.inference.sequence_generator_noilm import SequenceGeneratorWithInternalLanguageModel
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask


@register_task("speech_to_text_noilm")
class SpeechToTextNoILMTask(SpeechToTextTask):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.encoder_avg_outs = getattr(args, 'encoder_avg_outs', None)
        self.ilm_weight = getattr(args, 'ilm_weight', None)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechToTextTask.add_args(parser)
        parser.add_argument(
            "--encoder-avg-outs",
            metavar="D",
            help="location of encoder avgs",
        )
        parser.add_argument(
            "--ilm-weight",
            metavar="W",
            default=0.2,
            type=float,
            help="weight for the internal language model",
        )

    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
    ):
        assert len(models) == 1
        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["ilm_weight"] = self.ilm_weight
        extra_gen_cls_kwargs["encoder_avg_outs"] = self.encoder_avg_outs
        return super().build_generator(
            models,
            args,
            seq_gen_cls=SequenceGeneratorWithInternalLanguageModel,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs)

