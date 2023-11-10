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
from examples.speech_to_text.models.conformer import ConformerModel, conformer_base_architecture, conformer_s
from examples.speech_to_text.models.joint_ctc_multitask import JointCtcMultiTaskModel

from fairseq.models import register_model, register_model_architecture


@register_model('conformer_joint_ctc')
class JointCtcConformer(JointCtcMultiTaskModel):
    encoder_parent_model = ConformerModel

    @staticmethod
    def add_args(parser):
        ConformerModel.add_args(parser)
        JointCtcMultiTaskModel.add_args(parser)


@register_model_architecture('conformer_joint_ctc', 'conformer_joint_ctc')
def base_conformer_joint_ctc_architecture(args):
    conformer_base_architecture(args)


@register_model_architecture('conformer_joint_ctc', 'conformer_joint_ctc_s')
def conformer_joint_ctc_s(args):
    conformer_s(args)
