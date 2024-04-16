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
import unittest

import torch

from examples.speech_to_text.occlusion_explanation.normalizers.paired_min_max import PairedMinMaxNormalizer
from examples.speech_to_text.occlusion_explanation.normalizers.single_mean_std import SingleMeanStdNormalizer


class TestNormalizers(unittest.TestCase):
    def setUp(self) -> None:
        self.single_mean_std_normalizer = SingleMeanStdNormalizer()
        self.paired_min_max_normalizer = PairedMinMaxNormalizer()
        self.fbank_explanations = torch.tensor(
            [[[1., 2., 1., 3.],
              [1., 2., 1., 3.],
              [1., 2., 1., 3.]],
             [[1., 2., 1., 0.],
              [1., 2., 1., 0.],
              [1., 2., 1., 0.]],
             [[1., 2., 1., 2.],
              [1., 2., 1., 2.],
              [1., 2., 1., 2.]],
             [[1., 2., 1., 5.],
              [1., 2., 1., 5.],
              [1., 2., 1., 5.]]])
        self.tgt_explanations = torch.tensor(
            [[[1.], [0.], [0.], [0.]],
             [[1.], [2.], [0.], [0.]],
             [[0.], [1.], [2.], [0.]],
             [[2.], [0.], [1.], [1.]]])

    def test_single_mean_std_normalization(self):
        expected_fbank = torch.tensor(
            [[[-0.8660,  0.2887, -0.8660,  1.4434],
              [-0.8660,  0.2887, -0.8660,  1.4434],
              [-0.8660,  0.2887, -0.8660,  1.4434]],
             [[0., 1.3540, 0., -1.3540],
              [0., 1.3540, 0., -1.3540],
              [0., 1.3540, 0., -1.3540]],
             [[-0.9574,  0.9574, -0.9574,  0.9574],
              [-0.9574,  0.9574, -0.9574,  0.9574],
              [-0.9574,  0.9574, -0.9574,  0.9574]],
             [[-0.73, -0.146, -0.73,  1.6061],
              [-0.73, -0.146, -0.73,  1.6061],
              [-0.73, -0.146, -0.73,  1.6061]]])
        expected_tgt = torch.tensor(
            [[[0.], [0.], [0.], [0.]],
             [[-0.7071], [0.7071], [0.], [0.]],
             [[-1.], [0.], [1.], [0.]],
             [[1.2247], [-1.2247], [0.], [0.]]])
        norm_fbank, norm_tgt = self.single_mean_std_normalizer(self.fbank_explanations, self.tgt_explanations)
        print(norm_fbank)
        print(norm_tgt)
        self.assertTrue(torch.allclose(norm_fbank, expected_fbank, atol=0.0001))
        self.assertTrue(torch.allclose(norm_tgt, expected_tgt, atol=0.0001))

    def test_paired_min_max_normalization(self):
        expected_fbank = torch.tensor(
            [[[0., 0.5, 0., 1.],
              [0., 0.5, 0., 1.],
              [0., 0.5, 0., 1.]],
             [[0.5, 1., 0.5, 0.],
              [0.5, 1., 0.5, 0.],
              [0.5, 1., 0.5, 0.]],
             [[0.5, 1., 0.5, 1.],
              [0.5, 1., 0.5, 1.],
              [0.5, 1., 0.5, 1.]],
             [[0.2, 0.4, 0.2, 1.],
              [0.2, 0.4, 0.2, 1.],
              [0.2, 0.4, 0.2, 1.]]])
        expected_tgt = torch.tensor(
            [[[0.], [0.], [0.], [0.]],
             [[0.5], [1.], [0.], [0.]],
             [[0.], [0.5], [1.], [0.]],
             [[0.4], [0.], [0.2], [0.2]]])
        norm_fbank, norm_tgt = self.paired_min_max_normalizer(
            self.fbank_explanations, self.tgt_explanations)
        self.assertTrue(torch.equal(norm_fbank, expected_fbank))
        self.assertTrue(torch.equal(norm_tgt, expected_tgt))


if __name__ == '__main__':
    unittest.main()
