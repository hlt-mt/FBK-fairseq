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

from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc

from fairseq.data import Dictionary


class MockS2TDataConfigSrc(S2TDataConfigSrc):
    def __init__(self):
        self.config = {}


class BaseSpeechTestCase:

    def init_sample_dataset(self, ds_cls) -> None:
        self.src_dict = Dictionary()
        src_lines = ["I like quokkas", "I like tortoises", "I like elephants"]
        for l in src_lines:
            self.src_dict.encode_line(l)
        self.tgt_dict = Dictionary()
        tgt_lines = ["Mi piacciono i quokka", "Mi piacciono le tartarughe", "Mi piacciono gli elefanti"]
        for l in tgt_lines:
            self.tgt_dict.encode_line(l)
        self.ds = ds_cls(
            "quokka",
            True,
            MockS2TDataConfigSrc(),
            ["f1.wav", "f2.wav", "f3.wav"],
            [30, 100, 27],
            src_lines,
            tgt_lines,
            ["s1", "s2", "s3"],
            tgt_dict=self.tgt_dict,
            src_dict=self.src_dict,
        )
