# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import math
from collections import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class ExponentialWarmupLRSchedulerConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=50000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    alpha_lr: float = field(
        default=4.0,
        metadata={"help": "controls the slope of the warmup phase"},
    )
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("exponential_warmup", dataclass=ExponentialWarmupLRSchedulerConfig)
class ExponentialWarmupLRScheduler(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number after
    an exponential warmup.

    The slope of the exponential warmup can be controlled by the `alpha_lr` hyperparameter:
    the lower it is, the more the learning rate scheduler resembles a linear warmup.

    During warmup::

      lr = cfg.lr * (e ** (cfg.alpha_lr * step / cfg.warmup_updates) - 1) / (e ** cfg.alpha_lr - 1)

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, cfg: ExponentialWarmupLRSchedulerConfig, optimizer):
        super().__init__(cfg, optimizer)
        if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with inverse_sqrt."
                " Consider --lr-scheduler=fixed instead."
            )
        warmup_end_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr

        # precompute the step-wise increase of the exponent
        self.exponential_slope = cfg.alpha_lr / cfg.warmup_updates

        self.exponential_lr_step = warmup_end_lr / (math.exp(cfg.alpha_lr) - 1)

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * cfg.warmup_updates ** 0.5

        # initial learning rate
        self.lr = 0.0
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.cfg.warmup_updates:
            self.lr = (math.exp(self.exponential_slope * num_updates) - 1) * self.exponential_lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
