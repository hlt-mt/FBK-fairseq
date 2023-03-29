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
import math
import unittest
from argparse import Namespace

import torch
from torch import nn

from examples.speech_to_text.models.conformer import conformer_s, ConformerEncoder
from examples.speech_to_text.modules.conformer_attention import MultiHeadedSelfAttentionModule
from examples.speech_to_text.modules.conformer_encoder_layer import ConformerEncoderLayer
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

    def test_encoder_components(self):
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
        encoder_layer = ConformerEncoderLayer(batchnorm_args)
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
        encoder = ConformerEncoder(batchnorm_args, self.fake_dict)
        encoder.eval()
        net_out = encoder.forward(fake_sample, fake_lengths, return_all_hiddens=True)
        self.assertTrue(
            torch.all(net_out["encoder_out"][0][1, 13:, :] == 0.0),
            f"non-zero entries in {net_out['encoder_out'][0][1, 13:, :]}")

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
        att_mask = padding_mask.unsqueeze(1).repeat([1, fake_sample.shape[1], 1])
        att_mask = att_mask.logical_or(att_mask.transpose(1, 2))
        att_mask2 = padding_mask2.unsqueeze(1).repeat([1, fake_sample2.shape[1], 1])
        att_mask2 = att_mask2.logical_or(att_mask2.transpose(1, 2))
        attn = MultiHeadedSelfAttentionModule(8, 4)
        attn.eval()
        attn_out = attn(fake_sample, att_mask)
        attn_out2 = attn(fake_sample2, att_mask2)
        torch.testing.assert_allclose(attn_out[1, :3, :], attn_out2[0])
        self.assertTrue(
            torch.all(attn_out[1, 3:, :] == 0.0), f"non-zero entries in {attn_out[1, 3:, :]}")

    def test_encoder_batch(self):
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        batchnorm_args.encoder_embed_dim = 8
        batchnorm_args.input_feat_per_channel = 8
        batchnorm_args.encoder_layers = 3
        fake_sample = torch.rand(5, 27, 8)
        fake_sample[1, 13:, :] = 0
        fake_sample[2, 8:, :] = 0
        fake_sample[3, 8:, :] = 0
        fake_sample[4, 5:, :] = 0
        fake_lengths = torch.LongTensor([27, 13, 8, 8, 5])
        encoder = ConformerEncoder(batchnorm_args, self.fake_dict)
        encoder.eval()
        net_out = encoder.forward(fake_sample, fake_lengths, return_all_hiddens=True)

        def test_item(item_idx):
            item_len = fake_lengths[item_idx].item()
            item_out_len = math.ceil(item_len / 4)
            fake_sample2 = fake_sample[item_idx, :item_len, :]
            net_out2 = encoder.forward(
                fake_sample2.unsqueeze(0), fake_lengths[item_idx].unsqueeze(0), return_all_hiddens=True)
            torch.testing.assert_allclose(
                    net_out["encoder_out"][0][:item_out_len, item_idx, :],
                    net_out2["encoder_out"][0][:, 0, :])

        for i in range(5):
            test_item(i)

    def test_encoder_batch_unsafe_fails(self):
        batchnorm_args = copy.deepcopy(self.base_args)
        batchnorm_args.no_syncbatchnorm = True
        batchnorm_args.encoder_embed_dim = 8
        batchnorm_args.input_feat_per_channel = 8
        batchnorm_args.encoder_layers = 3
        batchnorm_args.batch_unsafe_relative_shift = True
        fake_sample = torch.rand(2, 27, 8)
        fake_sample[1, 13:, :] = 0
        fake_lengths = torch.LongTensor([27, 13])
        encoder = ConformerEncoder(batchnorm_args, self.fake_dict)
        encoder.eval()
        net_out = encoder.forward(fake_sample, fake_lengths, return_all_hiddens=True)
        fake_sample2 = fake_sample[1, :13, :]
        net_out2 = encoder.forward(fake_sample2.unsqueeze(0), fake_lengths[1].unsqueeze(0), return_all_hiddens=True)
        with self.assertRaises(AssertionError) as ae:
            torch.testing.assert_allclose(
                net_out["encoder_out"][0][:4, 1, :],
                net_out2["encoder_out"][0][:, 0, :])
        self.assertTrue("Tensor-likes are not close!" in str(ae.exception))


if __name__ == '__main__':
    unittest.main()
