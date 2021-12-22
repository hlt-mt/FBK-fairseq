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

import torch

from examples.speech_to_text.models.speechformer import SpeechformerEncoder, speechformer_s

from examples.linformer.linformer_src.modules.conv1d_compress import Conv1dCompressLayer
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask


class TestConvLayers(unittest.TestCase):
    def test_double_layers(self):
        encoder_embed_dim = 512
        compressed = 4
        compress_kernel_size = 8
        n_layers = 2
        freeze_compress = False

        compress_layer = Conv1dCompressLayer(
            encoder_embed_dim,
            compress_kernel_size,
            compression_factor=compressed,
            padding=compress_kernel_size,
            n_layers=n_layers,
            freeze_compress=freeze_compress,
        )

        self.assertEqual(len(compress_layer.conv_layers), 2)
        self.assertEqual(compress_layer.conv_layers[0].stride[0], 2)
        self.assertEqual(compress_layer.conv_layers[1].stride[0], 2)

    def test_single_layer(self):
        encoder_embed_dim = 512
        compressed = 4
        compress_kernel_size = 8
        n_layers = 1
        freeze_compress = False

        compress_layer = Conv1dCompressLayer(
            encoder_embed_dim,
            compress_kernel_size,
            compression_factor=compressed,
            padding=compress_kernel_size,
            n_layers=n_layers,
            freeze_compress=freeze_compress,
        )

        self.assertEqual(len(compress_layer.conv_layers), 1)
        self.assertEqual(compress_layer.conv_layers[0].stride[0], 4)

    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        speechformer_s(cls.base_args)
        cls.base_args.input_feat_per_channel = 5
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.transformer_after_compression = True
        cls.base_args.ctc_encoder_layer = 4
        cls.base_args.ctc_compress_strategy = "none"
        cls.base_args.criterion = "ctc_multi_loss"
        cls.base_args.encoder_embed_dim = 64
        cls.base_args.compressed = 4
        cls.base_args.compress_kernel_size = 8
        cls.base_args.compress_n_layers = 2
        cls.base_args.freeze_compress = False
        cls.fake_dict = Dictionary()

    def test_speechformer_compression(self):
        encoder = SpeechformerEncoder(self.base_args, self.fake_dict)

        compress_layer = encoder.build_speechformer_encoder_layer(self.base_args)

        self.assertEqual(len(compress_layer.shared_compress_layer[0].conv_layers), 2)
        self.assertEqual(compress_layer.shared_compress_layer[0].conv_layers[0].stride[0], 2)
        self.assertEqual(compress_layer.shared_compress_layer[0].conv_layers[1].stride[0], 2)

    def test_speechformer_non_shared_compression(self):
        new_args = copy.deepcopy(self.base_args)
        # test non-shared compression layer initialization
        new_args.shared_layer_kv_compressed = 0
        # test Conv1D compression with n_layers > 1
        new_args.compress_n_layers = 2
        encoder = SpeechformerEncoder(new_args, self.fake_dict)
        compress_layer = encoder.build_speechformer_encoder_layer(new_args)
        self.assertEqual(compress_layer.shared_compress_layer[0], None)
        compress_list = []
        for module_i in range(4):
            compress_list.append(encoder._modules["speechformer_layers"]._modules[f"{module_i}"].self_attn.compress_k
                                 .conv_layers)
        for compress_ in compress_list:
            self.assertEqual(len(compress_), 2)
            self.assertEqual(compress_[0].stride[0], 2)
            self.assertEqual(compress_[1].stride[0], 2)
        # test Conv1D compression with n_layers > 1
        new_args.compress_n_layers = 1
        encoder = SpeechformerEncoder(new_args, self.fake_dict)
        compress_layer = encoder.build_speechformer_encoder_layer(new_args)
        self.assertEqual(compress_layer.shared_compress_layer[0], None)
        compress_list = []
        for module_i in range(4):
            compress_list.append(encoder._modules["speechformer_layers"]._modules[f"{module_i}"].self_attn.compress_k
                                 .conv_layers)
        for compress_ in compress_list:
            self.assertEqual(len(compress_), 1)
            self.assertEqual(compress_[0].stride[0], 4)

    def test_assertion(self):
        """
        Testing odd numbers for 2 layers of Conv1D has to raise an error because the final compression factor cannot be
        reached using the same stride in the Conv1D. Ex.: with compression_factor = 5, the max same stride that can be
        used to build 2 layers of Conv1D is 2, reaching 2*2 = 4 as final compression factor.
        """
        encoder_embed_dim = 512
        compressed = 5
        compress_kernel_size = 8
        n_layers = 2
        freeze_compress = False

        with self.assertRaises(AssertionError):
            Conv1dCompressLayer(
                encoder_embed_dim,
                compress_kernel_size,
                compression_factor=compressed,
                padding=compress_kernel_size,
                n_layers=n_layers,
                freeze_compress=freeze_compress,
            )

    def test_padding_mask(self):
        for n_layers in [1, 2]:
            new_args = copy.deepcopy(self.base_args)
            new_args.compress_n_layers = n_layers
            encoder = SpeechformerEncoder(new_args, self.fake_dict)
            encoder_layer = encoder.build_speechformer_encoder_layer(new_args)
            x = torch.rand((40, 8, 64))
            key_padding_mask = lengths_to_padding_mask(torch.LongTensor([40, 37, 31, 28, 23, 22, 4, 1]))
            out, attn_w = encoder_layer.self_attn(x, None, None, key_padding_mask=key_padding_mask)
            self.assertEqual(attn_w.shape, (8, 40, 11))
            for b in range(8):
                for i in range(40):
                    self.assertAlmostEqual(1.0, attn_w[b, i, :].sum().item(), places=5)

            start_padding_idxs_per_batch = {
                2: 9,  # 31
                3: 8,  # 28
                4: 7,  # 23
                5: 7,  # 22
                6: 2,  # 4
                7: 2  # 1
            }
            for b_idx in range(8):
                for out_idx in range(40):
                    for s_idx in range(11):
                        if s_idx >= start_padding_idxs_per_batch.get(b_idx, 11):
                            self.assertAlmostEqual(
                                0.0, attn_w[b_idx, out_idx, s_idx].item(), places=5,
                                msg=f"Attention at [{b_idx}, {out_idx}, {s_idx}] was not padded")
                        else:
                            self.assertGreater(
                                attn_w[b_idx, out_idx, s_idx].item(), 0.000001,
                                msg=f"Attention at [{b_idx}, {out_idx}, {s_idx}] was padded although not necessary")


if __name__ == '__main__':
    unittest.main()
