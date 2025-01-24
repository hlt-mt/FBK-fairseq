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

from collections import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class PieceWiseWarmupLRSchedulerConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=50000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    intermediate_warmup_updates: int = field(
        default=25000,
        metadata={"help": "first linear warmup of the learning rate happens for this many updates"},
    )
    warmup_init_lr: float = field(
        default=0,
        metadata={
            "help": "initial learning rate during warmup phase; default is 0"
        },
    )
    intermediate_lr: float = field(
        default=0,
        metadata={
            "help": "learning rate during the intermediate phase; must be <lr and >warmup_init_lr"
        },
    )
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("piecewise_warmup", dataclass=PieceWiseWarmupLRSchedulerConfig)
class PieceWiseWarmupLRScheduler(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number after
    a two-stage linear warmup (https://arxiv.org/pdf/2401.16658).

    We linearly increase from ``--warmup-init-lr`` to ``--intermediate-lr``
    for ``--intermediate-warmup-updates`` and then the learning rate is linearly
    increased up to the configured peak learning rate ``--lr`` at ``--warmup-updates``.
    Thereafter, we decay proportional to the number of updates, with a decay factor
    set to align with the configured learning rate.

    During the first warmup::

      lrs = torch.linspace(cfg.warmup_init_lr, cfg.intermediate_lr, cfg.intermediate_warmup_updates)
      lr = lrs[update_num]

    During the second warmup::

      lrs = torch.linspace(cfg.intermediate_lr, cfg.lr, cfg.warmup_updates - cfg.intermediate_warmup_updates)
      lr = lrs[update_num - cfg.intermediate_warmup_updates]

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, cfg: PieceWiseWarmupLRSchedulerConfig, optimizer):
        super().__init__(cfg, optimizer)
        if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with inverse_sqrt."
                " Consider --lr-scheduler=fixed instead."
            )
        warmup_end_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr

        if cfg.intermediate_lr <= cfg.warmup_init_lr or cfg.intermediate_lr >= warmup_end_lr:
            raise ValueError(
                "Cannot use an intermediate learning rate higher than the final lr or smaller than "
                "the warmup init_lr")

        # two linear warmups for the first cfg.warmup_updates
        self.first_lr_step = (cfg.intermediate_lr - cfg.warmup_init_lr) / \
            cfg.intermediate_warmup_updates

        self.second_lr_step = (warmup_end_lr - cfg.intermediate_lr) / \
            (cfg.warmup_updates - cfg.intermediate_warmup_updates)

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * cfg.warmup_updates ** 0.5

        # initial learning rate
        self.lr = cfg.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.cfg.intermediate_warmup_updates:
            self.lr = self.cfg.warmup_init_lr + num_updates * self.first_lr_step
        elif num_updates < self.cfg.warmup_updates:
            second_phase_updates = num_updates - self.cfg.intermediate_warmup_updates
            self.lr = self.cfg.intermediate_lr + second_phase_updates * self.second_lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
