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

from examples.speech_to_text.occlusion_explanation.aggregators.utils import _min_max_normalization, \
    _mean_std_normalization


class TestNormalization(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_min_max_normalization(self):
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
        norm_fbank, norm_tgt = _min_max_normalization(
            self.fbank_explanations, self.tgt_explanations)
        self.assertTrue(torch.equal(norm_fbank, expected_fbank))
        self.assertTrue(torch.equal(norm_tgt, expected_tgt))

    def test_mean_std_normalization(self):
        expected_fbank = torch.tensor(
            [[[-0.8429, 0.3746, -0.8429, 1.5922],
              [-0.8429, 0.3746, -0.8429, 1.5922],
              [-0.8429, 0.3746, -0.8429, 1.5922]],
             [[-0.1015, 1.3199, -0.1015, -1.5230],
              [-0.1015, 1.3199, -0.1015, -1.5230],
              [-0.1015, 1.3199, -0.1015, -1.5230]],
             [[-0.6547, 0.9820, -0.6547, 0.9820],
              [-0.6547, 0.9820, -0.6547, 0.9820],
              [-0.6547, 0.9820, -0.6547, 0.9820]],
             [[-0.6010, 0.0401, -0.6010, 1.9631],
              [-0.6010, 0.0401, -0.6010, 1.9631],
              [-0.6010, 0.0401, -0.6010, 1.9631]]])
        expected_tgt = torch.tensor(
            [[[-0.8429], [0.], [0.], [0.]],
             [[-0.1015], [1.3199], [0.], [0.]],
             [[-2.2913], [-0.6547], [0.9820], [0.]],
             [[0.0401], [-1.2420], [-0.6010], [-0.6010]]])
        norm_fbank, norm_tgt = _mean_std_normalization(
            self.fbank_explanations, self.tgt_explanations)
        self.assertTrue(torch.allclose(norm_fbank, expected_fbank, atol=0.001))
        self.assertTrue(torch.allclose(norm_tgt, expected_tgt, atol=0.001))


if __name__ == '__main__':
    unittest.main()
