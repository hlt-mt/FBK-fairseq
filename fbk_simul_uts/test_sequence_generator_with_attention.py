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
import unittest

import torch

from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel
from fairseq.sequence_generator import SequenceGenerator
from tests.test_sequence_generator import get_dummy_task_and_parser, get_dummy_dictionary


class SequenceGeneratorWithAttentionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task, cls.parser = get_dummy_task_and_parser()
        cls.dictionary = get_dummy_dictionary()
        cls.tgt_dict = cls.dictionary
        # B x T x C
        src_tokens = torch.rand(2, 24, 5)
        src_lengths = torch.LongTensor([2, 24])
        cls.sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}
        }
        S2TTransformerModel.add_args(cls.parser)
        args = cls.parser.parse_args([])
        args.encoder_layers = 2
        args.decoder_layers = 4
        args.input_feat_per_channel = 5
        args.input_channels = 1
        args.max_source_positions = 15
        args.max_target_positions = 20
        args.criterion = "label_smoothed_cross_entropy"
        args.ctc_compress_strategy = "none"
        cls.model = S2TTransformerModel.build_model(args, cls.task)
        cls.generator = SequenceGenerator([cls.model], cls.tgt_dict, beam_size=1)

    def test_sequence_generator_with_attention(self):
        hypos = self.generator.generate(
            self.model, sample=self.sample, prefix_tokens=None, constraints=None, bos_token=None,
            pre_computed_encoder_outs=None, extract_attn_from_layer=0
        )
        attn_scores = hypos[0][0]["attention"]
        # attention scores must be of dimension
        # T/4 (24 subsampled by a factor of 4 = 6) x max_target_positions (20)
        self.assertTrue(attn_scores.shape == (6, 20))


if __name__ == '__main__':
    unittest.main()
