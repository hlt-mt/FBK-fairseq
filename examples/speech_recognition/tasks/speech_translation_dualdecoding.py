# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from examples.speech_recognition.tasks.speech_translation_ctc import SpeechTranslationCTCTask
from examples.speech_recognition.twophase_sequence_generator import TwoPhaseSequenceGenerator
from fairseq import search
from fairseq.tasks import register_task


logger = logging.getLogger(__name__)


@register_task("speech_translation_dualdecoding")
class SpeechTranslationDualDecodingTask(SpeechTranslationCTCTask):
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

