# Copyright 2022 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from examples.speech_to_text.models.conformer import ConformerModel, conformer_base_architecture, conformer_s
from examples.speech_to_text.models.transformer_decoder_with_tags import TransformerDecoderWithTags
from fairseq.models import (
    register_model,
    register_model_architecture,
)


@register_model("conformer_with_tags")
class ConformerWithTagsModel(ConformerModel):
    """
    Adds support to predict additional tags to Conformer model.
    """
    @staticmethod
    def add_args(parser):
        """Adds model-specific arguments to the parser."""
        ConformerModel.add_args(parser)
        parser.add_argument('--add-tags-embeddings', default=False, action='store_true',
                            help='if set, the previous token embeddings are summed with embeddings of their tag')

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderWithTags(args, task.target_dictionary, embed_tokens, task.data_cfg.tags)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, return_all_hiddens=True)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            prev_target_tags=kwargs.get('prev_target_tags', None)
        )
        if self.encoder.ctc_flag:
            return decoder_out, {"ctc_out": encoder_out["ctc_out"], "ctc_lengths": encoder_out["ctc_lengths"]}
        else:
            return decoder_out


@register_model_architecture(model_name="conformer_with_tags", arch_name="conformer_with_tags")
def conformer_with_tags_base_architecture(args):
    conformer_base_architecture(args)


@register_model_architecture("conformer_with_tags", "conformer_with_tags_s")
def conformer_with_tags_s(args):
    conformer_s(args)
