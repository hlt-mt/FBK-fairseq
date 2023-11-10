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
from typing import Tuple

import torch
from torch.nn import functional as F

from examples.speech_to_text.inference.ctc_generator import CTCGenerator
from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task("speech_to_text_ctcgen_aux")
class SpeechToTextCtcGenAuxiliaryDecoderTask(SpeechToTextCtcTask):
    """
    Task for generating the output with the CTC decoder of a multitask model.
    """

    def build_generator(self, models, args, **kwargs):
        return CTCAuxiliaryDecoderGenerator(args, self.target_dictionary)


class CTCAuxiliaryDecoderGenerator(CTCGenerator):
    """
    Generates the output of the CTC auxiliary decoder of a multitask model.
    """

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.prefix_size = args.prefix_size

    def gen_logprobs_and_lengths(self, model, sample_input) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_input = {
            k: v for k, v in sample_input.items() if k != "prev_output_tokens"
        }
        encoder_out = model.encoder(**encoder_input, return_all_hiddens=False)
        lang_embeds = None
        if self.prefix_size > 0:
            assert self.prefix_size == 1, "prefix_size > 1 is not supported"
            # the prev_output_tokens contain <bos> <lang> and then the sentence
            # so the 2nd column corresponds to the language embeddings
            lang_embeds = model.decoder.embed_tokens(sample_input["prev_output_tokens"][:, 1:2])
        auxiliary_out = model.auxiliary_decoder(encoder_out, lang_embeds=lang_embeds)
        ctc_logprobs = F.log_softmax(auxiliary_out[0], dim=-1).to("cpu")
        ctc_logprobs = ctc_logprobs.transpose(0, 1)  # from T x B x C to B x T x C
        ctc_lengths = model.get_auxiliary_input_lens(sample_input, auxiliary_out)
        return ctc_logprobs, ctc_lengths
