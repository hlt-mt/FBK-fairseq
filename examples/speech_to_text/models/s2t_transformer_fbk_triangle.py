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
from examples.speech_to_text.models.base_triangle import BaseTriangle
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, base_architecture, \
    s2t_transformer_m, s2t_transformer_s
from fairseq.models import register_model, register_model_architecture


@register_model('s2t_transformer_triangle')
class S2TTransformerTriangle(BaseTriangle):
    encoder_parent_model = S2TTransformerModel

    @staticmethod
    def add_args(parser):
        BaseTriangle.add_args(parser)
        S2TTransformerModel.add_args(parser)

    @staticmethod
    def add_base_args(args):
        base_s2t_transformer_triangle_architecture(args)


@register_model_architecture('s2t_transformer_triangle', 's2t_transformer_triangle')
def base_s2t_transformer_triangle_architecture(args):
    base_architecture(args)


@register_model_architecture('s2t_transformer_triangle', 's2t_transformer_triangle_s')
def s2t_transformer_triangle_s(args):
    s2t_transformer_s(args)


@register_model_architecture('s2t_transformer_triangle', 's2t_transformer_triangle_m')
def s2t_transformer_triangle_m(args):
    s2t_transformer_m(args)
