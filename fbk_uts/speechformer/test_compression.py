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

from examples.speech_to_text.models.speechformer import speechformer_s, SpeechformerEncoder
from examples.speech_to_text.modules.ctc_support import CtcSupport
from fairseq.data import Dictionary


class SpeechformerCompressionTestCase(unittest.TestCase):
    base_args = None
    fake_dict = None

    fake_ctc_out = torch.Tensor([
        [[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [1., 0., 0., 0., 0.]],
        [[0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.]],
        [[0., 0., 1., 0., 0.], [1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 0., 1., 0., 0.]],
        [[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [1., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 1.], [1., 0., 0., 0., 0.], [0., 0., 0., 0., 1.], [0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 1.], [0., 1., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
        [[0., 0., 0., 1., 0.], [1., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
        [[0., 0., 0., 1., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
    ])
    fake_x = torch.Tensor([
        [[1., 1., 1., 1., 1.], [1.1, 1.1, 1.1, 1.1, 1.1], [1.2, 1.2, 1.2, 1.2, 1.2], [1.3, 1.3, 1.3, 1.3, 1.3]],
        [[2., 2., 2., 2., 2.], [2.1, 2.1, 2.1, 2.1, 2.1], [2.2, 2.2, 2.2, 2.2, 2.2], [2.3, 2.3, 2.3, 2.3, 2.3]],
        [[3., 3., 3., 3., 3.], [3.1, 3.1, 3.1, 3.1, 3.1], [3.2, 3.2, 3.2, 3.2, 3.2], [3.3, 3.3, 3.3, 3.3, 3.3]],
        [[4., 4., 4., 4., 4.], [4.1, 4.1, 4.1, 4.1, 4.1], [4.2, 4.2, 4.2, 4.2, 4.2], [0., 0., 0., 0., 0.]],
        [[5., 5., 5., 5., 5.], [5.1, 5.1, 5.1, 5.1, 5.1], [5.2, 5.2, 5.2, 5.2, 5.2], [0., 0., 0., 0., 0.]],
        [[6., 6., 6., 6., 6.], [6.1, 6.1, 6.1, 6.1, 6.1], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
        [[7., 7., 7., 7., 7.], [7.1, 7.1, 7.1, 7.1, 7.1], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
        [[8., 8., 8., 8., 8.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
    ])
    fake_x_lens = torch.LongTensor([8, 7, 5, 3])

    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        speechformer_s(cls.base_args)
        cls.base_args.input_feat_per_channel = 5
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.transformer_after_compression = True
        cls.base_args.ctc_encoder_layer = 4
        cls.base_args.ctc_compress_strategy = "avg"
        cls.base_args.criterion = "ctc_multi_loss"
        cls.base_args.ctc_compress_max_out_size = -1
        cls.base_args.ctc_compress_fixed_ratio = 4
        cls.fake_dict = Dictionary()

    def test_base_compression(self):
        encoder = SpeechformerEncoder(self.base_args, self.fake_dict)

        out_x, out_lens = encoder.average_same_ctc_features(self.fake_ctc_out, self.fake_x, self.fake_x_lens)
        self.assertEqual(out_lens[0].item(), 6)
        self.assertEqual(out_lens[1].item(), 7)
        self.assertEqual(out_lens[2].item(), 3)
        self.assertEqual(out_lens[3].item(), 3)

    def test_compression_with_maxout(self):
        custom_args = copy.deepcopy(self.base_args)
        custom_args.ctc_compress_max_out_size = 6
        encoder = SpeechformerEncoder(custom_args, self.fake_dict)

        out_x, out_lens = encoder.average_same_ctc_features(self.fake_ctc_out, self.fake_x, self.fake_x_lens)
        self.assertEqual(out_lens[0].item(), 6)
        # compressed by 2
        self.assertEqual(out_lens[1].item(), 4)
        self.assertEqual(out_lens[2].item(), 3)
        self.assertEqual(out_lens[3].item(), 3)

        custom_args.ctc_compress_max_out_size = 4
        encoder = SpeechformerEncoder(custom_args, self.fake_dict)
        out_x, out_lens = encoder.average_same_ctc_features(self.fake_ctc_out, self.fake_x, self.fake_x_lens)
        # compressed by 2
        self.assertEqual(out_lens[0].item(), 3)
        # compressed by 2
        self.assertEqual(out_lens[1].item(), 4)
        self.assertEqual(out_lens[2].item(), 3)
        self.assertEqual(out_lens[3].item(), 3)

        custom_args.ctc_compress_max_out_size = 3
        encoder = SpeechformerEncoder(custom_args, self.fake_dict)
        out_x, out_lens = encoder.average_same_ctc_features(self.fake_ctc_out, self.fake_x, self.fake_x_lens)
        # compressed by 2
        self.assertEqual(out_lens[0].item(), 3)
        # compressed by 3
        self.assertEqual(out_lens[1].item(), 3)
        self.assertEqual(out_lens[2].item(), 3)
        self.assertEqual(out_lens[3].item(), 3)
        fake_batch_predicted = [
            [('a', 1), ('b', 2), ('a', 2), ('c', 1), ('b', 2), ('a', 1), ('c', 5)],
            [('a', 10)]]
        out_batch_pred = encoder.ensure_max_ctc_out_len(fake_batch_predicted)
        self.assertEqual([('a', 5), ('b', 4), ('c', 5)], out_batch_pred[0])
        self.assertEqual([('a', 10)], out_batch_pred[1])

    def test_fixed_compression(self):
        custom_args = copy.deepcopy(self.base_args)
        custom_args.ctc_compress_strategy = "fixed"
        encoder = SpeechformerEncoder(custom_args, self.fake_dict)

        out_x, out_lens = encoder.average_same_ctc_features(self.fake_ctc_out, self.fake_x, self.fake_x_lens)
        self.assertEqual(out_lens[0].item(), 2)
        self.assertEqual(out_lens[1].item(), 2)
        self.assertEqual(out_lens[2].item(), 2)
        self.assertEqual(out_lens[3].item(), 1)
        for i in range(5):
            self.assertAlmostEqual(out_x[0][3].tolist()[i], 2.3, places=5)
        self.assertEqual(out_x[1][3].tolist(), [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_fixed_compression_8(self):
        custom_args = copy.deepcopy(self.base_args)
        custom_args.ctc_compress_strategy = "fixed"
        custom_args.ctc_compress_fixed_ratio = 8
        encoder = SpeechformerEncoder(custom_args, self.fake_dict)
        self.assertEqual(CtcSupport.FIXED_RATIO, 8)

        out_x, out_lens = encoder.average_same_ctc_features(self.fake_ctc_out, self.fake_x, self.fake_x_lens)
        self.assertEqual(out_lens[0].item(), 1)
        self.assertEqual(out_lens[1].item(), 1)
        self.assertEqual(out_lens[2].item(), 1)
        self.assertEqual(out_lens[3].item(), 1)
        for i in range(5):
            self.assertAlmostEqual(out_x[0][3].tolist()[i], 2.3, places=5)


if __name__ == '__main__':
    unittest.main()
