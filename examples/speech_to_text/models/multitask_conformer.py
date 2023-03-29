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
from examples.speech_to_text.models.multi_task import MultiTaskClassifierModel
from fairseq.models import register_model, register_model_architecture


@register_model('multitask_conformer')
class MultitaskConformer(MultiTaskClassifierModel):
    @staticmethod
    def add_args(parser):
        ConformerModel.add_args(parser)
        MultiTaskClassifierModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        base_model = ConformerModel.build_model(args, task)
        return cls.build_with_classifier(base_model, args, task)


@register_model_architecture('multitask_conformer', 'multitask_conformer')
def conformer_multitask_base_architecture(args):
    conformer_base_architecture(args)


@register_model_architecture('multitask_conformer', 'multitask_conformer_s')
def conformer_multitask_s(args):
    conformer_s(args)
