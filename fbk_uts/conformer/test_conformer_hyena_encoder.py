# Copyright 2024 FBK

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

from torch import nn, Tensor, LongTensor

from examples.speech_to_text.models.conformer_hyena import conformer_hyena_s, ConformerHyenaEncoder
from examples.speech_to_text.modules.conformer_hyena_encoder_layer import ConformerHyenaEncoderLayer
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask

from pangolinn import seq2seq


class ConformerHyenaEncoderLayerPangolinnWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    def build_module(self) -> nn.Module:
        base_args = Namespace()
        base_args.input_feat_per_channel = self.num_input_channels
        base_args.input_channels = 1
        base_args.max_source_positions = 300
        base_args.no_syncbatchnorm = True
        base_args.encoder_embed_dim = 8
        base_args.stride = 1
        conformer_hyena_s(base_args)
        return ConformerHyenaEncoderLayer(base_args)

    @property
    def num_input_channels(self) -> int:
        return 8

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self._module(x.transpose(0, 1), lengths_to_padding_mask(lengths)).transpose(0, 1)


class ConformerHyenaEncoderPangolinnWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    def base_args(self) -> Namespace:
        base_args = Namespace()
        base_args.input_feat_per_channel = self.num_input_channels
        base_args.input_channels = 1
        base_args.max_source_positions = 300
        base_args.no_syncbatchnorm = True
        base_args.encoder_embed_dim = 8
        base_args.encoder_layers = 3
        base_args.stride = 1
        base_args.criterion = "ctc_multi_loss"
        base_args.ctc_compress_strategy = "none"
        base_args.ctc_encoder_layer = 2
        conformer_hyena_s(base_args)
        return base_args

    def build_module(self) -> nn.Module:
        return ConformerHyenaEncoder(self.base_args(), Dictionary())

    @property
    def num_input_channels(self) -> int:
        return 8

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self._module(x, lengths)["encoder_out"][0].transpose(0, 1)


class ConformerHyenaEncoderLayerPaddingTestCase(seq2seq.EncoderPaddingTestCase):
    module_wrapper_class = ConformerHyenaEncoderLayerPangolinnWrapper


class ConformerHyenaEncoderPaddingTestCase(seq2seq.EncoderPaddingTestCase):
    module_wrapper_class = ConformerHyenaEncoderPangolinnWrapper


class ConformerHyenaEncoderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        cls.base_args.input_feat_per_channel = 5
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 300
        cls.base_args.criterion = "ctc_multi_loss"
        cls.base_args.ctc_encoder_layer = 4
        cls.base_args.ctc_compress_strategy = "none"
        cls.base_args.stride = 1
        conformer_hyena_s(cls.base_args)
        cls.fake_dict = Dictionary()

    def test_encoder_components(self):
        encoder = ConformerHyenaEncoder(self.base_args, self.fake_dict)

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
        encoder = ConformerHyenaEncoder(args, self.fake_dict)
        for layer in range(len(encoder._modules["conformer_layers"])):
            self.assertTrue(
                isinstance(encoder._modules["conformer_layers"][layer].conv_module.batchnorm, norm_class))


if __name__ == '__main__':
    unittest.main()
