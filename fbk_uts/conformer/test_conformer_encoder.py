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

from examples.speech_to_text.modules.conformer_encoder_layer import ConformerEncoderLayer
from torch import nn, rand, all, LongTensor

from examples.speech_to_text.models.conformer import conformer_s, ConformerEncoder
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask


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

    def test_syncbatchnorm(self):
        # Test case in which no_syncbatchnorm is False (default)
        self.check_norm(self.base_args, nn.SyncBatchNorm)
        # Test case in which no_syncbatchnorm is True
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        self.check_norm(batchnorm_args, nn.BatchNorm1d)

    def check_norm(self, args, norm_class):
        encoder = ConformerEncoder(args, self.fake_dict)
        for layer in range(len(encoder._modules["conformer_layers"])):
            isinstance(encoder._modules["conformer_layers"][layer].conv_module.batchnorm, norm_class)

    def test_conformer_convolutional_layer_padding(self):
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        batchnorm_args.encoder_embed_dim = 8
        fake_sample = rand(2, 10, 8)
        fake_sample[1, 3:, :] = 0
        fake_lengths = LongTensor([10, 3])
        padding_mask = lengths_to_padding_mask(fake_lengths)
        encoder_layer = ConformerEncoderLayer(batchnorm_args)
        encoder_layer.eval()
        out = encoder_layer.conv_module(fake_sample, padding_mask).transpose(0, 1)
        self.assertTrue(all(out[1, 3:, :] == 0.0), f"non-zero entries in {out[1, 3:, :]}")
