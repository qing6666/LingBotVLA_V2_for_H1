# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
lr_scheduler.py —— 学习率调度器
====================================================================
【作用】控制训练过程中学习率(lr)怎么随 step 变化。先 warmup(线性升温,防开局训崩),
       再按所选风格衰减/保持。

【四种风格(lr_decay_style)】
  constant   :warmup 后保持不变                  ← LingBot-VLA 2.0 默认
  linear     :warmup 后线性降到 min_lr
  cosine     :warmup 后余弦降到 min_lr(最平滑)
  two_stage  :warmup → 恒定 → 到 decay_steps 降到 decay_ratio(OpenVLA-OFT 风格)

【机制】都基于 torch 的 LambdaLR:传一个 _lr_lambda(step)→ 比例(0~1),
       实际 lr = init_lr × 比例。

【被谁用】train_lingbotvla.py: lr_scheduler = build_lr_scheduler(optimizer, ...)
====================================================================
"""

import math
from typing import TYPE_CHECKING, Literal

from torch.optim.lr_scheduler import LambdaLR

from ..utils import logging


if TYPE_CHECKING:
    from torch.optim import Optimizer


logger = logging.get_logger(__name__)


def build_lr_scheduler(
    optimizer: "Optimizer",
    train_steps: int,
    lr: float = 1e-3,
    lr_decay_style: Literal["constant", "linear", "cosine", "two_stage"] = "constant",
    lr_decay_ratio: float = 1.0,
    lr_warmup_ratio: float = 0.0,
    lr_min: float = 1e-7,
    lr_start: float = 0.0,
):
    """★ 入口:按 lr_decay_style 选调度策略。warmup 步数 = train_steps × lr_warmup_ratio。"""
    lr_warmup_steps = int(train_steps * lr_warmup_ratio)
    if lr_decay_style == "constant":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            lr_start=lr_start,
            init_lr=lr,
        )

    if lr_decay_style == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_steps,
            init_lr=lr,
            lr_start=lr_start,
        )

    if lr_decay_style == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_steps,
            init_lr=lr,
            lr_decay_ratio=lr_decay_ratio,
            min_lr=lr_min,
            lr_start=lr_start,
        )

    if lr_decay_style == "two_stage":
        return get_two_stage_constant_schedule_with_warmup(
            optimizer=optimizer,
            init_lr=lr,
            lr_start=lr_start,
        )

    raise ValueError(f"Unknown learning rate decay style: {lr_decay_style}.")


def get_constant_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    恒定:warmup 线性升温到 init_lr,之后保持不变。
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # warmup:从 lr_start 线性升到 init_lr
            return (lr_start + (init_lr - lr_start) * current_step / max(1, num_warmup_steps)) / init_lr

        return 1.0   # warmup 后:保持 init_lr(比例=1)

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)

def get_two_stage_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int = 100,
    init_lr: float = 0.0,
    decay_steps: int = 10_000,
    decay_ratio: float = 0.2,
    last_epoch: int = -1,
    lr_start: float = 0.0,
):
    """
    Two stages constant learning rate schedule with warm up follows OpenVLA-OFT.
    Defaultly, the learning rate is 0.0 for the first 100 steps and then decay to 0.2 * init_lr after 10_000 steps.
    两阶段(OpenVLA-OFT):warmup → 恒定 → 到 decay_steps 降到 decay_ratio×init_lr。
    """
    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            warmup_lr = lr_start + (init_lr - lr_start) * (current_step / max(1, num_warmup_steps))
            return warmup_lr / init_lr

        if current_step >= decay_steps:
            return decay_ratio   # 第二阶段:降到 decay_ratio(默认 0.2)

        return 1.0   # 第一阶段:保持 init_lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    线性衰减:warmup 升到 init_lr,之后线性降到 min_lr。
    """
    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * current_step / max(1, num_warmup_steps)) / init_lr

        min_lr_ratio = min_lr / init_lr
        # warmup 后:从 1 线性降到 min_lr_ratio(不低于 min_lr)
        return max(
            min_lr_ratio,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    lr_decay_ratio: float = 1.0,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases following the values of the cosine function between
    the initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0
    and the initial lr set in the optimizer.
    余弦衰减:warmup 升到 init_lr,之后按余弦曲线平滑降到 min_lr(比 linear 更平滑)。
    """
    def lr_lambda(current_step: int):
        lr_decay_steps = int(num_training_steps * lr_decay_ratio)
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * current_step / max(1, num_warmup_steps)) / init_lr

        min_lr_ratio = min_lr / init_lr
        if current_step > lr_decay_steps:
            return min_lr_ratio   # 超过衰减步数:保持 min_lr

        progress = float(current_step - num_warmup_steps) / float(max(1, lr_decay_steps - num_warmup_steps))
        assert 0 <= progress <= 1
        # 余弦因子:从 1 平滑降到 0,再缩放到 [min_lr_ratio, 1]
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1 - min_lr_ratio) + min_lr_ratio
        return max(0, factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
