# Copyright 2023 FBK

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
from torch.nn import functional as F

from examples.speech_to_text.modules.gradient_reversal import gradient_reversal, GradientReversalLayer


class GRLTestCase(unittest.TestCase):
    def test_basic(self):
        a = torch.Tensor([1, 2])
        w = torch.nn.Parameter(torch.Tensor([2]))
        x = gradient_reversal(a * w, torch.tensor(1.0, requires_grad=False))
        y = F.l1_loss(x, torch.Tensor([1, 2]))
        y.backward()
        self.assertEqual(w.grad[0], -1.5)

    def test_after_grl(self):
        a = torch.Tensor([1, 2])
        w = torch.nn.Parameter(torch.Tensor([2]))
        x = gradient_reversal(a, torch.tensor(1.0, requires_grad=False)) * w
        y = F.l1_loss(x, torch.Tensor([1, 2]))
        y.backward()
        self.assertEqual(w.grad[0], 1.5)

    def test_no_lambda_gamma_set(self):
        with self.assertRaises(
                AssertionError, msg="either lambda_factor or gamma should not be None"):
            _ = GradientReversalLayer()

    def test_max_updates_with_gamma(self):
        with self.assertRaises(
                AssertionError,
                msg="max_updates should be a positive number when adaptive factor is used"):
            _ = GradientReversalLayer(gamma=10)
        with self.assertRaises(
                AssertionError,
                msg="max_updates should be a positive number when adaptive factor is used"):
            _ = GradientReversalLayer(gamma=10, max_updates=0)

    def test_gamma(self):
        grl = GradientReversalLayer(gamma=10, max_updates=100)
        self.assertAlmostEqual(grl._lambda_factor.item(), 0)
        grl.set_num_updates(1000)
        self.assertAlmostEqual(grl._lambda_factor.item(), 1.0)

    def test_lambda(self):
        grl = GradientReversalLayer(gamma=10, max_updates=100, lambda_factor=2)
        self.assertAlmostEqual(grl._lambda_factor.item(), 2.0)
        grl.set_num_updates(1000)
        self.assertAlmostEqual(grl._lambda_factor.item(), 2.0)


if __name__ == '__main__':
    unittest.main()
