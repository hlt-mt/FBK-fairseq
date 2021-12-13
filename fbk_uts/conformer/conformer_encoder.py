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
from argparse import Namespace

from examples.speech_to_text.models.conformer import conformer_s, ConformerEncoder
from fairseq.data import Dictionary


class ConformerEncoderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        cls.base_args.input_feat_per_channel = 5
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.criterion = "ctc_multi_loss"
        cls.base_args.ctc_encoder_layer = 4
        cls.base_args.ctc_compress_strategy = "none"
        conformer_s(cls.base_args)
        cls.fake_dict = Dictionary()

    def test_encoder(self):
        encoder = ConformerEncoder(self.base_args, self.fake_dict)

        correct_components = ["dropout_module", "subsample", "conformer_layers", "ctc_fc"]
        self.assertListEqual(list(encoder.__dict__["_modules"].keys()), correct_components)