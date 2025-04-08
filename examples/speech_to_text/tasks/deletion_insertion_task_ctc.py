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
from examples.speech_to_text.tasks.deletion_insertion_support import FeatureAttributionEvaluationSupport
from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from fairseq.tasks import register_task


LOGGER = logging.getLogger(__name__)


@register_task("feature_attribution_evaluation_task_ctc")
class FeatureAttributionEvaluationCtcTask(SpeechToTextCtcTask, FeatureAttributionEvaluationSupport):
    @staticmethod
    def add_args(parser):
        SpeechToTextCtcTask.add_args(parser)
        FeatureAttributionEvaluationSupport.add_args(parser)

    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        self.setup_evaluation(args)
