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

from examples.linformer.linformer_src.modules.conv1d_compress import Conv1dCompressLayer


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


if __name__ == '__main__':
    unittest.main()
