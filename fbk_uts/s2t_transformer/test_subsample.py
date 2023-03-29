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

import torch

from examples.speech_to_text.models.conformer import conformer_s
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerEncoder
from fairseq.data import Dictionary


class S2TTransformerEncoderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        cls.base_args.input_feat_per_channel = 5
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.criterion = "ctc_multi_loss"
        cls.base_args.ctc_encoder_layer = 2
        cls.base_args.encoder_layers = 2
        cls.base_args.ctc_compress_strategy = "none"
        cls.base_args.encoder_embed_dim = 8
        conformer_s(cls.base_args)
        cls.fake_dict = Dictionary()

    def test_subsample_with_padding(self):
        encoder = S2TTransformerEncoder(self.base_args, self.fake_dict)
        encoder.eval()
        torch.manual_seed(0)
        fake_sample = torch.rand(2, 40, 5)
        fake_sample[1, 20:, :] = 0
        fake_lengths = torch.LongTensor([40, 20])
        fake_sample2 = fake_sample[1:, :20, :]
        subsample_out = encoder.subsample(fake_sample, fake_lengths)
        subsample_out2 = encoder.subsample(fake_sample2, fake_lengths[1:])
        # Sum of padded elements must be equal to 0
        self.assertTrue((subsample_out[0][5:, 1, :].sum() == 0).item())
        # Non padded elements of the processed batch must be equal to the unpadded sample
        torch.testing.assert_allclose(
                subsample_out[0][:5, 1, :],
                subsample_out2[0][:, 0, :])
        encoder_out = encoder(fake_sample, fake_lengths)
        encoder_out2 = encoder(fake_sample2, fake_lengths[1:])
        # Ensure encoder output
        torch.testing.assert_allclose(
                encoder_out['encoder_out'][0][:5, 1, :],
                encoder_out2['encoder_out'][0][:, 0, :])


if __name__ == '__main__':
    unittest.main()
