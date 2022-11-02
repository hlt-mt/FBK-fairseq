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
import math

from dataclasses import dataclass

from api.speech_processor import SpeechToTextProcessor
from fairseq import utils
from fairseq_cli.generate import get_symbols_to_strip_from_output


@dataclass
class STProcessorRequest:
    """
    A data class that represents a request to be processed by the
    :py:class:`STProcessor`.
    """
    wav_path: str
    src_lang: str
    tgt_lang: str


@dataclass
class STProcessorResponse:
    """
    A data class that represents a response produced by the
    :py:class:`STProcessor`.
    """
    score: float
    translation: str


class STProcessor(SpeechToTextProcessor):
    request_class = STProcessorRequest

    def __init__(self, cfg):
        super().__init__(cfg)

    def _postproc(self, request_id, hypo, request: STProcessorRequest):
        hypo_tokens = hypo['tokens'].int().cpu()
        self.logger.info(
            f"Inference for Request ID[{request_id}] generated "
            f"{len(hypo_tokens)} target tokens.")
        hypo_confidence = utils.item(hypo['score'] / math.log(2))

        detok_hypo_str = self._postproc_out_and_tags(
            request_id,
            hypo_tokens,
            self.tgt_dict,
            hypo["tags"] if "tags" in hypo else None,
            get_symbols_to_strip_from_output(self.generator),
        )
        return STProcessorResponse(hypo_confidence, detok_hypo_str)
