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
from typing import Optional, Dict, Any, List

from torch import nn

from examples.speech_to_text.data.speech_to_text_dataset_tagged import S2TDataConfigTagged
from examples.speech_to_text.models.base_triangle_with_prev_tags import TriangleTransformerDecoderWithTags, \
    TransformerDecoderWithTags, BaseTrianglePreviousTags
from examples.speech_to_text.models.s2t_transformer_fbk_triangle import S2TTransformerTriangle
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, base_architecture, \
    s2t_transformer_m, s2t_transformer_s
from fairseq.models import register_model, register_model_architecture


@register_model('s2t_transformer_triangle_with_tags')
class S2TTransformerTriangleWithTags(BaseTrianglePreviousTags):
    """
    This model is an implementation of a multi-task model that predicts both transcripts
    and translations, with the translation being generated from the output representation
    of the transcript decoder. It represents the triangle model of (Sperber et al. 2020).
    In addition, it outputs for each token in both transcripts and translations
    the corresponding tag information (e.g. the named entity type).
    """
    encoder_parent_model = S2TTransformerModel

    @staticmethod
    def add_args(parser):
        BaseTrianglePreviousTags.add_args(parser)
        S2TTransformerModel.add_args(parser)


@register_model_architecture('s2t_transformer_triangle_with_tags', 's2t_transformer_triangle_with_tags')
def base_triangle_with_tags_architecture(args):
    base_architecture(args)


@register_model_architecture('s2t_transformer_triangle_with_tags', 's2t_transformer_triangle_with_tags_s')
def s2t_triangle_with_tags_s(args):
    s2t_transformer_s(args)


@register_model_architecture('s2t_transformer_triangle_with_tags', 's2t_transformer_triangle_with_tags_m')
def s2t_triangle_with_tags_m(args):
    s2t_transformer_m(args)
