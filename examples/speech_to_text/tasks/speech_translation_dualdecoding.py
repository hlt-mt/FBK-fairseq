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

from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from examples.speech_to_text.inference.twophase_sequence_generator import TwoPhaseSequenceGenerator
from fairseq.tasks import register_task


logger = logging.getLogger(__name__)


@register_task("speech_translation_dualdecoding")
class SpeechTranslationDualDecodingTask(SpeechToTextCtcTask):
    """
    Task for training dual-decoder models for joint speech translation and recognition.
    """

    def build_generator(self, models, args):
        return TwoPhaseSequenceGenerator(
            models,
            self.source_dictionary,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
        )

