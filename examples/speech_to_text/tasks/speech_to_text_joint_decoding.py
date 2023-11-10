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
import logging

from examples.speech_to_text.inference.sequence_generator_joint_ctc_decoding import SequenceGeneratorJointCTCDecoding, SequenceGeneratorJointCTCDecodingWithAlignment
from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task("speech_to_text_joint_decoding")
class SpeechToTextCtcAuxiliaryJointDecodingTask(SpeechToTextCtcTask):
    """
    Task for generating the output with joint decoding of the CTC decoder and
    autoregressive decoder of a multitask model.
    """
    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        self.ctc_weight = getattr(args, 'ctc_decode_weight', None)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechToTextCtcTask.add_args(parser)
        parser.add_argument(
            "--ctc-decode-weight",
            metavar="W",
            default=0.2,
            type=float,
            help="weight of CTC rescoring",
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
        extra_gen_cls_kwargs["ctc_weight"] = self.ctc_weight
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorJointCTCDecodingWithAlignment
                extra_gen_cls_kwargs['print_alignment'] = args.print_alignment
                extra_gen_cls_kwargs['extract_attn_from_layer'] = self.extract_attn_from_layer
            else:
                seq_gen_cls = SequenceGeneratorJointCTCDecoding
        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs)
