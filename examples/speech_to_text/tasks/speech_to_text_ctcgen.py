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
from typing import Tuple

import torch
from torch.nn import functional as F

from examples.speech_to_text.inference.ctc_generator import CTCGenerator
from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from fairseq.data import BaseWrapperDataset
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task("speech_to_text_ctcgen")
class SpeechToTextCtcGenTask(SpeechToTextCtcTask):
    """
    Task for generating the transcripts with the CTC module of an encoder-decoder model.
    """
    use_src_dict_as_target = True

    def build_generator(self, models, args, **kwargs):
        return CTCSourceGenerator(args, self.source_dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)
        # We need to remove the target from the returned dataset,
        # as in this scenario the target is the translation into another language
        # but when we decode the CTC output we are generating the transcript
        # in the source language.
        # As such, we use the source dictionary in the generate for the target,
        # so if we do not remove the target here, it would be printed, generating
        # it with the wrong (source) dictionary, potentially causing also exceptions.
        self.datasets[split] = MaskTargetDataset(self.datasets[split])


class MaskTargetDataset(BaseWrapperDataset):
    def collater(self, samples):
        sample = super().collater(samples)
        sample["target"] = None
        return sample


class CTCSourceGenerator(CTCGenerator):
    """
    Generates the output of a CTC multi loss operator, where the CTC is
    computed on the source side.
    """
    def gen_logprobs_and_lengths(self, model, sample_input) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_input = {
            k: v for k, v in sample_input.items() if k != "prev_output_tokens"
        }
        encoder_out = model.encoder(**encoder_input, return_all_hiddens=True)
        ctc_features = encoder_out["ctc_out"]
        ctc_logprobs = F.log_softmax(ctc_features, dim=-1).to("cpu")
        ctc_lengths = encoder_out["ctc_lengths"]
        if not getattr(ctc_features, "batch_first", False):
            ctc_logprobs = ctc_logprobs.transpose(0, 1)
        return ctc_logprobs, ctc_lengths
