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
import copy
import unittest
from argparse import Namespace

from examples.speech_to_text.models.speechformer import speechformer_s, SpeechformerEncoder, speechformer_hybrid_s
from examples.speech_to_text.modules.conformer_encoder_layer import ConformerEncoderLayer
from examples.speech_to_text.modules.speechformer_encoder_layer import SpeechformerEncoderLayer
from fairseq.data import Dictionary
from fairseq.modules import TransformerEncoderLayer


class SpeechformerTestCase(unittest.TestCase):
    base_args = None

    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        cls.base_args.encoder_embed_dim = 16
        cls.base_args.input_feat_per_channel = 5
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.ctc_encoder_layer = 4
        cls.base_args.encoder_layers = 8
        cls.base_args.ctc_compress_strategy = "avg"
        cls.base_args.criterion = "ctc_multi_loss"
        cls.base_args.ctc_compress_max_out_size = -1
        speechformer_s(cls.base_args)
        speechformer_hybrid_s(cls.base_args)
        cls.fake_dict = Dictionary()

    def do_check(self, additional_args, expected_class):
        args = copy.deepcopy(self.base_args)
        for k, v in additional_args:
            args.__setattr__(k, v)
        encoder = SpeechformerEncoder(args, self.fake_dict)
        for i in range(4):
            self.assertIsInstance(encoder.speechformer_layers[i], SpeechformerEncoderLayer)
        for i in range(4, 8):
            self.assertIsInstance(encoder.speechformer_layers[i], expected_class)

    def test_speechformer(self):
        self.do_check([("transformer_after_compression", True)], TransformerEncoderLayer)

    def test_speechformer_plus_conformer(self):
        self.do_check([("conformer_after_compression", True)], ConformerEncoderLayer)

    def test_failure(self):
        try:
            self.do_check(
                [("transformer_after_compression", True), ("conformer_after_compression", True)],
                ConformerEncoderLayer)
        except AssertionError as e:
            self.assertEqual(str(e), "Cannot enable both transformer_after_compression and conformer_after_compression")


if __name__ == '__main__':
    unittest.main()
