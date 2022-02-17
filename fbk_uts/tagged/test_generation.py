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
import unittest

import torch

from examples.speech_to_text.generate_dualdecoder import join_tags_tokens
from fairseq.data import Dictionary


class GenerationTestCase(unittest.TestCase):
    def do_test_tags_token_joiner(self, tokens, tags_list, tag_logprobs, expected_tags_string, expected_out_string):
        d = Dictionary()
        for t in tags_list:
            d.add_symbol(f"<{t}>")
            d.add_symbol(f"</{t}>")
        for t in tokens:
            d.add_symbol(t)
        d.finalize()
        tokens_encoded = torch.IntTensor([d.index(t) for t in tokens])
        tags_strings, joint_string = join_tags_tokens(tag_logprobs, tokens_encoded, d, tags_list)
        self.assertEqual(expected_tags_string, ' '.join(tags_strings))
        self.assertEqual(expected_out_string, d.string(joint_string))

    def test_conflicting_tags(self):
        tag_logprobs = torch.log(torch.Tensor([
            [0.90, 0.01, 0.09, 0.00],  # \u2581This
            [0.87, 0.05, 0.03, 0.05],  # \u2581is
            [0.97, 0.01, 0.01, 0.01],  # \u2581a
            [0.10, 0.31, 0.41, 0.18],  # \u2581quo
            [0.08, 0.60, 0.27, 0.05],  # k
            [0.10, 0.39, 0.41, 0.10],  # ka
            [0.97, 0.01, 0.01, 0.01],  # in
            [0.33, 0.11, 0.42, 0.14],  # \u2581Tren
            [0.10, 0.31, 0.41, 0.18],  # to
        ]))
        self.do_test_tags_token_joiner(
            ["\u2581This", "\u2581is", "\u2581a", "\u2581quo", "k", "ka", "\u2581in", "\u2581Tren", "to"],
            ["A", "B", "C"],
            tag_logprobs,
            "- - - A A A - B B",
            "\u2581This \u2581is \u2581a <A> \u2581quo k ka </A> \u2581in <B> \u2581Tren to </B>"
        )

    def test_punctuation(self):
        tag_logprobs = torch.log(torch.Tensor([
            [0.90, 0.01, 0.09, 0.00],  # \u2581This
            [0.87, 0.05, 0.03, 0.05],  # \u2581is
            [0.97, 0.01, 0.01, 0.01],  # \u2581a
            [0.10, 0.31, 0.41, 0.18],  # \u2581quo
            [0.08, 0.60, 0.27, 0.05],  # k
            [0.10, 0.39, 0.41, 0.10],  # ka
            [0.97, 0.01, 0.01, 0.01],  # ,
            [0.97, 0.01, 0.01, 0.01],  # in
            [0.33, 0.11, 0.42, 0.14],  # \u2581Tren
            [0.10, 0.31, 0.41, 0.18],  # to
            [0.97, 0.01, 0.01, 0.01],  # .
        ]))
        self.do_test_tags_token_joiner(
            ["\u2581This", "\u2581is", "\u2581a", "\u2581quo", "k", "ka", ",", "\u2581in", "\u2581Tren", "to", "."],
            ["A", "B", "C"],
            tag_logprobs,
            "- - - A A A - - B B -",
            "\u2581This \u2581is \u2581a <A> \u2581quo k ka </A> , \u2581in <B> \u2581Tren to </B> ."
        )

    def test_consecutive_tagged_words(self):
        tag_logprobs = torch.log(torch.Tensor([
            [0.90, 0.01, 0.09, 0.00],  # \u2581This
            [0.87, 0.05, 0.03, 0.05],  # \u2581is
            [0.08, 0.60, 0.27, 0.05],  # \u2581a
            [0.10, 0.31, 0.41, 0.18],  # \u2581quo
            [0.08, 0.60, 0.27, 0.05],  # k
            [0.10, 0.39, 0.41, 0.10],  # ka
            [0.33, 0.11, 0.42, 0.14],  # \u2581Tren
            [0.10, 0.31, 0.41, 0.18],  # tino
            [0.97, 0.01, 0.01, 0.01],  # .
        ]))
        self.do_test_tags_token_joiner(
            ["\u2581This", "\u2581is", "\u2581a", "\u2581quo", "k", "ka", "\u2581Tren", "tino", "."],
            ["A", "B", "C"],
            tag_logprobs,
            "- - A A A A B B -",
            "\u2581This \u2581is <A> \u2581a \u2581quo k ka </A> <B> \u2581Tren tino </B> ."
        )


if __name__ == '__main__':
    unittest.main()
