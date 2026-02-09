# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig

# global states for scheduling
# these are needed as LambdaLR does not support argument passing
_warmup_steps = 200
_decay_steps = 0


def linear_warmup_linear_decay(current_step: int) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    Final learning rate will be 10% of the initial learning rate.
    """
    if current_step < _warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (_warmup_steps + 1))
    else:
        # linear decay from 1.0 to 0.1 (10% of initial lr)
        
        decay_ratio = (current_step - _warmup_steps) / _decay_steps
        if not 0 <= decay_ratio <= 1:
            print(f"decay_ratio = {decay_ratio} is not in the range [0, 1]. Please check the warmup_steps and steps.")
        # Linear interpolation from 1.0 to 0.1
        curr_adjustment = 1.0 - decay_ratio * (1.0 - 0.1)
        curr_adjustment = max(0.1, curr_adjustment)  # Ensure minimum is 0.1

    return curr_adjustment


def linear_warmup_cosine_decay(it: int) -> float:
    # print(it, _warmup_steps, _decay_steps)
    '''A cosine decay schedule with warmup. Return a multiplicative factor to adjust the learning rate. Note the input is 1-indexed.'''
    # 1) linear warmup for warmup_iters steps
    if it < _warmup_steps:
        return it / _warmup_steps
    # use cosine decay down to min learning rate
    decay_ratio = (it - _warmup_steps) / _decay_steps
    if not 0 <= decay_ratio <= 1:
        print(it, _warmup_steps, _decay_steps)
        print(f"decay_ratio = {decay_ratio} is not in the range [0, 1]. Please check the warmup_steps and steps.")
        print(f"decay_ratio = {decay_ratio} is not in the range [0, 1]. Please check the warmup_steps and steps.")
        print(f"decay_ratio = {decay_ratio} is not in the range [0, 1]. Please check the warmup_steps and steps.")
    import math
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return 0.1 + coeff * (1 - 0.1)


def get_lr_schedulers(optimizers, job_config: JobConfig, scheduler_type="linear"):
    def _get_lr_scheduler(optimizer):
        """Build a learning rate scheduler with warmup and decay"""
        global _warmup_steps, _decay_steps
        _warmup_steps = int(job_config.training.warmup_steps)
        _decay_steps = float(max(1, job_config.training.steps - _warmup_steps))
        
        if scheduler_type == "linear":
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_linear_decay)
        else:  # default to cosine
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_cosine_decay)
        
        return warmup_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for schedulers in self.schedulers:
                schedulers.step()

    return SchedulersContainer(
        [_get_lr_scheduler(optimizer) for optimizer in optimizers]
    )


if __name__ == "__main__":
    _warmup_steps = 20
    _decay_steps = 100

    import torch
    optimizer = torch.optim.AdamW([torch.zeros(1)], lr=1)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_cosine_decay)
    for i in range(1, 101):
        optimizer.step()
        warmup_scheduler.step()
        print(i, warmup_scheduler.get_last_lr())
