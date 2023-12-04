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

import argparse
import unittest

import torch
from fairseq import search
from fairseq.sequence_generator import SequenceGenerator
from tests.utils import TestTranslationTask, dummy_dictionary


class TestBeamSearchAvoidRepeatedEobEol(unittest.TestCase):
    def setUp(self):
        # construct dummy dictionary
        d = dummy_dictionary(vocab_size=2)
        self.eob = d.add_symbol("<eob>")
        self.eol = d.add_symbol("<eol>")

        self.eos = d.eos()
        self.w1 = 4
        self.w2 = 5

        # construct source data
        src_tokens = torch.LongTensor([[self.w1, self.w2, self.eos], [self.w1, self.w2, self.eos]])
        src_lengths = torch.LongTensor([2, 2])

        args = argparse.Namespace()
        unk = 0.0
        args.beam_probs = [
            # step 0:
            torch.FloatTensor(
                [
                    # eos      w1   w2   eob  eol
                    # sentence 1:
                    # test we avoid starting with eob/eol (even if they are likely)
                    [0.0, unk, 0.2, 0.1, 0.6, 0.1],  # beam 1
                    [0.0, unk, 0.2, 0.1, 0.1, 0.6],  # beam 2
                    # sentence 2:
                    [0.0, unk, 0.6, 0.2, 0.1, 0.1],
                    [0.0, unk, 0.6, 0.2, 0.1, 0.1],
                ]
            ),
            # step 1:
            torch.FloatTensor(
                [
                    # sentence 1:
                    [1.0, unk, 0.0, 0.0, 0.0, 0.0],  # w1: 0.9  (emit: w1 <eos>: 0.9*1.0)
                    [0.0, unk, 0.1, 0.1, 0.8, 0.0],  # eob has to be predicted now
                    # sentence 2:
                    [0.25, unk, 0.35, 0.4, 0.0, 0.0],  # w1: 0.7  (don't emit: w1 <eos>: 0.7*0.25)
                    [0.00, unk, 0.10, 0.0, 0.9, 0.0],  # w2: 0.3
                ]
            ),
            # step 2:
            torch.FloatTensor(
                [
                    # sentence 1:
                    [0.0, unk, 0.15, 0.05, 0.8, 0.0],  # eob after eob should not be allowed
                    [0.0, unk, 0.1, 0.1, 0.0, 0.8],  # eol after eob should not be allowed
                    # sentence 2:
                    [0.6, unk, 0.4, 0.0, 0.0, 0.0],  # w1 w2: 0.7*0.4  (emit: w1 w2 <eos>: 0.7*0.4*0.6)
                    [0.01, unk, 0.0, 0.99, 0.0, 0.0],  # w2 w2: 0.3*0.9
                ]
            ),
            # step 3:
            # finish
            torch.FloatTensor(
                [
                    # sentence 1:
                    [1.0, unk, 0.0, 0.0, 0.0, 0.0],
                    [1.0, unk, 0.0, 0.0, 0.0, 0.0],
                    # sentence 2:
                    [0.1, unk, 0.5, 0.4, 0.0, 0.0],
                    [1.0, unk, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ]

        task = TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary
        self.sample = {"net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}}

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertListEqual(t1.tolist(), t2.tolist())

    def generate(self, avoid_repeated_eob_eol=False):
        if avoid_repeated_eob_eol:
            search_strategy = search.BeamSearchNoRepeatedEobEol(self.tgt_dict)
        else:
            search_strategy = None
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        return generator.forward(self.sample)

    def test_base_beam_search(self):
        hypos = self.generate()
        # sentence 1, beam 1
        self.assertTensorEqual(hypos[0][0]["tokens"], torch.LongTensor([self.eob, self.eos]))
        # sentence 1, beam 2
        self.assertTensorEqual(
            hypos[0][1]["tokens"], torch.LongTensor([self.w1, self.eob, self.eob, self.eos]))
        # sentence 2, beam 1
        self.assertTensorEqual(
            hypos[1][0]["tokens"], torch.LongTensor([self.w1, self.w2, self.w1, self.eos]))
        # sentence 2, beam 2
        self.assertTensorEqual(
            hypos[1][1]["tokens"], torch.LongTensor([self.w1, self.w2, self.eos]))

    def test_beam_search_avoid_repeated(self):
        hypos = self.generate(avoid_repeated_eob_eol=True)

        # sentence 1, beam 1
        self.assertTensorEqual(hypos[0][0]["tokens"], torch.LongTensor([self.w1, self.eos]))
        # sentence 1, beam 2
        self.assertTensorEqual(
            hypos[0][1]["tokens"], torch.LongTensor([self.w2, self.eob, self.w1, self.eos]))
        # sentence 2, beam 1
        self.assertTensorEqual(
            hypos[1][0]["tokens"], torch.LongTensor([self.w1, self.w2, self.w1, self.eos]))
        # sentence 2, beam 2
        self.assertTensorEqual(
            hypos[1][1]["tokens"], torch.LongTensor([self.w1, self.w2, self.eos]))


if __name__ == "__main__":
    unittest.main()
