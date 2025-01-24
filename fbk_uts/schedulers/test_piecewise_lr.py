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
from argparse import Namespace

import torch

from fairseq import optim
from fairseq.optim.lr_scheduler.piecewise_warmup_scheduler import PieceWiseWarmupLRSchedulerConfig, \
    PieceWiseWarmupLRScheduler


class TestPiecewiseLRScheduler(unittest.TestCase):
    def test_piecewise_lr(self):
        linear = torch.nn.Linear(2, 2)
        args = Namespace()
        args.lr = [0.1]
        args.momentum = 0.1
        args.weight_decay = 0.1
        opt = optim.sgd.SGD(args, linear.parameters())
        config = PieceWiseWarmupLRSchedulerConfig(intermediate_lr=0.01, lr=[0.1])
        lr_scheduler = PieceWiseWarmupLRScheduler(config, opt)
        lr = lr_scheduler.step_update(0)
        self.assertAlmostEqual(lr, 0.)
        lr = lr_scheduler.step_update(1000)
        self.assertAlmostEqual(lr, 0.0004)
        lr = lr_scheduler.step_update(25000)
        self.assertAlmostEqual(lr, 0.01)
        lr = lr_scheduler.step_update(30000)
        self.assertAlmostEqual(lr, 0.028)
        lr = lr_scheduler.step_update(50000)
        self.assertAlmostEqual(lr, 0.1)
        lr = lr_scheduler.step_update(100000)
        self.assertAlmostEqual(lr, 0.07071067811865475)


if __name__ == '__main__':
    unittest.main()
