# Copyright 2024 FBK

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

from examples.speech_to_text.models.conformer import ConformerModel, \
    conformer_base_architecture, conformer_s
from examples.speech_to_text.models.s2tlm_transformer_fbk import AudioPrependedTransformerDecoder
from fairseq.models import register_model, register_model_architecture


logger = logging.getLogger(__name__)


@register_model("s2tlm_conformer")
class S2TLMConformerModel(ConformerModel):
    @staticmethod
    def add_args(parser):
        ConformerModel.add_args(parser)
        parser.add_argument(
            "--causal-prompt-mask", action="store_true", default=False,
            help="Apply causual masking on audio prompt"
        )

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return AudioPrependedTransformerDecoder(args, task.target_dictionary, embed_tokens, no_encoder_attn=True)


@register_model_architecture(model_name="s2tlm_conformer", arch_name="s2tlm_conformer")
def s2tlm_conformer_base_architecture(args):
    conformer_base_architecture(args)


@register_model_architecture("s2tlm_conformer", "s2tlm_conformer_s")
def s2tlm_transformer_s(args):
    conformer_s(args)
