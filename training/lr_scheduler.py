"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import math
from torch.optim.lr_scheduler import _LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR


def get_lr_scheduler(scheduler, optimizer, warmup_epochs, total_epochs):
    """scheduler: config中定义的参数"""
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=1E-3,
                                    total_iters=warmup_epochs)

        if scheduler == 'PolynomialLRWithWarmup':
            decay_scheduler = PolynomialLR(optimizer,
                                        total_iters=total_epochs-warmup_epochs,
                                        power=1)
        elif scheduler in 'CosineAnnealingLRWithWarmup':
            decay_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs-warmup_epochs, eta_min=1E-8)
        else:
            raise NotImplementedError

        return SequentialLR(optimizer,
                            schedulers=[warmup_scheduler, decay_scheduler],
                            milestones=[warmup_epochs])  # 有两种scheduler
    else:
        return CosineAnnealingLR(optimizer, T_max=total_epochs-warmup_epochs, eta_min=1E-8)


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, power, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) /
                        (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warm-up phase: linearly increase learning rate for each parameter group
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs if base_lr != 0 else 0
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase: use cosine function for decay for each parameter group
            cos_decay = 0.5 * (
                1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))
            )
            return [
                base_lr * cos_decay if base_lr != 0 else 0
                for base_lr in self.base_lrs
            ]
