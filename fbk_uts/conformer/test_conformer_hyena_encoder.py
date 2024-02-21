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

import torch
from torch import nn

from examples.speech_to_text.models.conformer_hyena import conformer_hyena_s, ConformerHyenaEncoder
from examples.speech_to_text.modules.conformer_hyena_encoder_layer import ConformerHyenaEncoderLayer
from examples.speech_to_text.modules.hyena import HyenaOperator
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask


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

    def test_conformer_encoder_layer_padding(self):
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        batchnorm_args.encoder_embed_dim = 8
        fake_sample = torch.rand(2, 10, 8)
        fake_sample[1, 3:, :] = 0
        fake_lengths = torch.LongTensor([10, 3])
        padding_mask = lengths_to_padding_mask(fake_lengths)
        encoder_layer = ConformerHyenaEncoderLayer(batchnorm_args)
        encoder_layer.eval()
        out = encoder_layer(fake_sample.transpose(0, 1), padding_mask).transpose(0, 1)
        self.assertTrue(
            torch.all(out[1, 3:, :] == 0.0), f"non-zero entries in {out[1, 3:, :]}")

    def test_encoder_padding(self):
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        batchnorm_args.encoder_embed_dim = 8
        batchnorm_args.input_feat_per_channel = 8
        batchnorm_args.encoder_layers = 3
        fake_sample = torch.rand(2, 27, 8)
        fake_sample[1, 13:, :] = 0
        fake_lengths = torch.LongTensor([27, 13])
        encoder = ConformerHyenaEncoder(batchnorm_args, self.fake_dict)
        encoder.eval()
        net_out = encoder.forward(fake_sample, fake_lengths, return_all_hiddens=True)
        padding_area = net_out["encoder_out"][0][13:, 1, :]  # output is N x B x C
        self.assertGreater(padding_area.numel(), 0)
        self.assertTrue(torch.all(padding_area == 0.0), f"non-zero entries in {padding_area}")

    def test_multihead_selfattn(self):
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        batchnorm_args.encoder_embed_dim = 8
        fake_sample = torch.rand(2, 10, 8)
        fake_sample[1, 3:, :] = 0
        fake_lengths = torch.LongTensor([10, 3])
        padding_mask = lengths_to_padding_mask(fake_lengths)
        fake_sample2 = fake_sample[1:, :3, :]
        padding_mask2 = lengths_to_padding_mask(fake_lengths[1].unsqueeze(0))
        attn = HyenaOperator(8, 10, num_heads=4)
        attn.eval()
        attn_out = attn(fake_sample, padding_mask)
        attn_out2 = attn(fake_sample2, padding_mask2)
        torch.testing.assert_allclose(attn_out[1, :3, :], attn_out2[0])
        self.assertTrue(
            torch.all(attn_out[1, 3:, :] == 0.0), f"non-zero entries in {attn_out[1, 3:, :]}")

    def test_encoder_batch(self):
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        batchnorm_args.encoder_embed_dim = 8
        batchnorm_args.input_feat_per_channel = 8
        batchnorm_args.encoder_layers = 1
        fake_sample = torch.rand(5, 27, 8)
        fake_sample[1, 13:, :] = 0
        fake_sample[2, 8:, :] = 0
        fake_sample[3, 8:, :] = 0
        fake_sample[4, 5:, :] = 0
        fake_lengths = torch.LongTensor([27, 13, 8, 8, 5])
        encoder = ConformerHyenaEncoder(batchnorm_args, self.fake_dict)
        encoder.eval()
        net_out = encoder.forward(fake_sample, fake_lengths, return_all_hiddens=True)

        def test_item(item_idx):
            item_len = fake_lengths[item_idx].item()
            fake_sample2 = fake_sample[item_idx, :item_len, :]
            net_out2 = encoder.forward(
                fake_sample2.unsqueeze(0), fake_lengths[item_idx].unsqueeze(0), return_all_hiddens=True)
            torch.testing.assert_allclose(
                    net_out["encoder_out"][0][:item_len, item_idx, :],
                    net_out2["encoder_out"][0][:, 0, :])

        for i in range(5):
            test_item(i)

    def test_not_looking_at_the_future(self):
        test_len = 20
        x = torch.rand(5, test_len, 8)
        batch_lens = torch.LongTensor([test_len] * 5)
        padding_mask = lengths_to_padding_mask(batch_lens)
        encoder = HyenaOperator(8, test_len, num_heads=4)
        output = encoder.forward(x, padding_mask)
        for j in range(19):
            # Checks that for each of the 20 elements we obtain the same prefix in the
            # results when feeding the model with the full input sequences and the input
            # prefix truncated at that element.
            partial_lens = torch.LongTensor([j + 1] * 5)
            partial_padding_mask = lengths_to_padding_mask(partial_lens)
            partial_output = encoder.forward(x[:, :j + 1, :], partial_padding_mask)
            torch.testing.assert_close(
                partial_output,
                output[:, :j + 1, :])

    def test_noncausal(self):
        test_len = 20
        x = torch.rand(5, test_len, 8)
        batch_lens = torch.LongTensor([test_len] * 5)
        padding_mask = lengths_to_padding_mask(batch_lens)
        encoder = HyenaOperator(8, test_len, num_heads=4, causal=False)
        output = encoder.forward(x, padding_mask)
        for j in range(19):
            # Checks that for each of the 20 elements we obtain the same prefix in the
            # results when feeding the model with the full input sequences and the input
            # prefix truncated at that element.
            partial_lens = torch.LongTensor([j + 1] * 5)
            partial_padding_mask = lengths_to_padding_mask(partial_lens)
            partial_output = encoder.forward(x[:, :j + 1, :], partial_padding_mask)
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(
                    partial_output,
                    output[:, :j + 1, :])


if __name__ == '__main__':
    unittest.main()
