# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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
optimizer.py —— 优化器工厂(构建各种优化器)
====================================================================
【作用】根据配置产出对应的优化器。三种选择:
  · adamw            标准 AdamW(torch 自带,可选 fused 加速)
  · anyprecision_adamw  低精度 AdamW(momentum/variance 用 bf16 + Kahan 求和,省显存)
  · muon             Muon 优化器(梯度正交化)→ 走 build_muon_optimizer

【关键组件】
  AnyPrecisionAdamW       低精度 AdamW 变体(省显存)
  build_optimizer         ★ 构建 adamw / anyprecision_adamw(含 depth 参数 lr_gain)
  CombinedOptimizer       组合器:把 Muon + AdamW 包成一个对外统一
  build_muon_optimizer    ★ 构建 Muon(2D 权重→Muon,1D→AdamW,组合成 CombinedOptimizer)

【被谁用】train_lingbotvla.py:
    optimizer = build_optimizer(...) 或 build_muon_optimizer(...)
====================================================================
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..utils.import_utils import is_torch_npu_available
from .muon import DistributedMuon, split_muon_adamw_params


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(Optimizer):
    """低精度 AdamW:momentum/variance 用低精度(bf16)省显存,用 Kahan 求和补偿精度损失。"""
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=True,
        momentum_dtype=torch.bfloat16,          # 动量用低精度省显存
        variance_dtype=torch.bfloat16,          # 方差用低精度省显存
        compensation_buffer_dtype=torch.bfloat16,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        一步更新(标准 AdamW 公式,但用低精度状态 + Kahan 求和)。

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values   动量(梯度的一阶矩)
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values   方差(二阶矩)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker   Kahan 误差补偿缓冲
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style   AdamW 式权重衰减
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum   更新动量
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance   更新方差

                bias_correction1 = 1 - beta1**step  # adjust using bias1   偏差校正
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation   用 Kahan 求和更新(补偿低精度误差)
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))   # 把舍入误差存回补偿缓冲
                else:  # usual AdamW updates   普通更新
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    post_training=False,
) -> "torch.optim.Optimizer":
    """★ 构建 AdamW / AnyPrecisionAdamW(muon 会报错引导到 build_muon_optimizer)。
    特别处理:若模型有 depth 相关参数(蒸馏分支),给它们单独一组,后训练时 lr×10(学得更快)。"""
    if param_groups is None:
        # 找出所有 depth 相关参数(蒸馏分支)
        align_parameters = [
            name for name, _ in model.named_parameters() if "depth" in name
        ]

        if len(align_parameters) > 0:
            lr_gain = 10.0 if not post_training else 1.0    # 后训练时 depth 分支 lr×10
            param_groups = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (p.requires_grad and n not in align_parameters)
                    ],
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if (p.requires_grad and n in align_parameters)
                    ],
                    "lr": lr * lr_gain,                      # depth 分支用更大 lr
                }
            ]
        else:
            param_groups = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type == "adamw":
        foreach = False if is_torch_npu_available() else (not fused)
        fused = False if is_torch_npu_available() else fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        optim = AnyPrecisionAdamW(param_groups, lr, betas, eps, weight_decay)
    elif optimizer_type == "muon":
        raise ValueError(
            "optimizer_type='muon' must go through build_muon_optimizer(model, args_train, ...) "
            "so that 1D params can be routed to AdamW."
        )   # muon 必须走 build_muon_optimizer(因为要分 1D/2D)
    else:
        raise ValueError("Only adamw and anyprecision_adamw are supported as optimizers.")

    return optim


class CombinedOptimizer(Optimizer):
    """Drive several inner optimizers as if they were a single one.
    ★ 组合器:把多个内部优化器(如 Muon + AdamW)包成一个,对外当单个优化器用。"""

    def __init__(self, optimizers: Sequence[Optimizer]):
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one inner optimizer.")
        self.optimizers: List[Optimizer] = list(optimizers)
        self.defaults = {}
        self._step_pre_hooks: List[Any] = []

    @property
    def param_groups(self):
        # 汇总所有内部优化器的 param_group(给 LR scheduler 用)
        groups: List[Dict[str, Any]] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        # LR schedulers mutate the shared param-group dicts; reassignment is a no-op.
        pass

    @property
    def state(self):
        # 合并所有内部优化器的 state(给 checkpoint 用)
        merged: Dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def register_step_pre_hook(self, hook):
        # 注册 pre-hook(给 MoE 负载均衡 hook 用)到第一个内部优化器
        return self.optimizers[0].register_step_pre_hook(hook)

    def step(self, closure=None):
        # ★ 依次调用每个内部优化器的 step(Muon 先,AdamW 后)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # checkpoint:把各内部优化器的 state 分别存
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def _split_param_groups_by_scaled_lr(
    params_and_names: Sequence[Tuple[torch.Tensor, str]],
    base_lr: float,
    layer_to_scale: Dict[int, float],
    layer_re: "re.Pattern[str]",
) -> List[Dict[str, Any]]:
    """Bucket (param, name) pairs by their possibly MoE-scaled LR.
    按"是否被 MoE 专家 lr 缩放"把参数分到不同 lr 桶。"""
    lr_to_params: Dict[float, List[torch.Tensor]] = {base_lr: []}
    for p, name in params_and_names:
        m = layer_re.search(name)
        lr_for_param = base_lr
        if m is not None:
            layer_idx = int(m.group(1))
            scale = layer_to_scale.get(layer_idx)
            if scale is not None:
                lr_for_param = base_lr * scale           # 该层是 MoE 专家 → 用放大 lr
        lr_to_params.setdefault(lr_for_param, []).append(p)
    return [{"params": ps, "lr": lr} for lr, ps in lr_to_params.items() if ps]


def build_muon_optimizer(
    model: "nn.Module",
    args_train,
    lr: float,
    weight_decay: float = 0.0,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
) -> "torch.optim.Optimizer":
    """Build DistributedMuon for matrix-like weights plus AdamW fallback groups.
    ★ 构建 Muon:2D 权重→DistributedMuon(正交化),1D(embedding/norm)→AdamW,组合成 CombinedOptimizer。"""
    # ① 分组:2D 权重(muon_params)/ 1D 等(adamw_params)
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        no_decay_modules=None,
        no_decay_params=None,
        extra_adamw_name_patterns=getattr(args_train, "muon_exclude_name_patterns", None) or None,
    )

    # ② MoE 专家 lr 缩放(use_moe_expert_lr 时,专家用放大 lr)
    use_expert_lr = bool(getattr(args_train, "use_moe", False)) and bool(
        getattr(args_train, "use_moe_expert_lr", False)
    )
    layer_to_scale: Dict[int, float] = {}
    layer_re = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.")
    if use_expert_lr:
        token_moe_layers = set(getattr(args_train, "token_moe_layers", None) or [])
        if token_moe_layers:
            token_scale = (args_train.token_num_experts / args_train.token_top_k) ** 0.5   # scale=(E/k)^0.5
            for idx in token_moe_layers:
                layer_to_scale[idx] = token_scale

    # ③ 各自按 expert lr 再分桶
    muon_groups = _split_param_groups_by_scaled_lr(
        list(zip(muon_params, muon_names)), lr, layer_to_scale, layer_re
    )
    adamw_groups = _split_param_groups_by_scaled_lr(
        list(zip(adamw_params, adamw_names)), lr, layer_to_scale, layer_re
    )

    if not muon_groups:
        raise RuntimeError(
            "build_muon_optimizer: no Muon-eligible (2D/3D) parameters were found. "
            "Use build_optimizer(optimizer_type='adamw') instead."
        )

    # ④ 建 Muon 优化器(2D 权重)
    muon_opt = DistributedMuon(
        muon_groups,
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(getattr(args_train, "muon_momentum", 0.95)),
        nesterov=bool(getattr(args_train, "muon_nesterov", True)),
        ns_steps=int(getattr(args_train, "muon_ns_steps", 5)),
        adjust_lr_fn=getattr(args_train, "muon_adjust_lr_fn", "match_rms_adamw"),
    )

    # ⑤ 建 AdamW(1D 参数),和 Muon 组合成 CombinedOptimizer
    inner_opts: List[Optimizer] = [muon_opt]
    if adamw_groups:
        foreach = not is_torch_npu_available()
        adamw_opt = AdamW(
            adamw_groups,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            fused=False,
            foreach=foreach,
        )
        inner_opts.append(adamw_opt)

    return CombinedOptimizer(inner_opts)   # 对外返回一个组合优化器
