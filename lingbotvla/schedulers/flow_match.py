# Copyright 2023 Zhongjie Duan
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
flow_match.py —— Flow Matching 噪声调度器(数学公式封装)
====================================================================
【作用】把 Flow Matching 的核心数学(加噪 / 目标 / 去噪)封装成一个调度器。
       管理 sigma(噪声水平)时间表,提供训练和推理要用的几个公式。

【核心概念】sigma(σ)= "噪声比例 / 浑浊度"
   σ=1 → 纯噪声;σ=0 → 纯动作。σ 从 1 递减到 0 就是"从噪声澄清到动作"。

【四个核心公式(对应之前讲的 flow matching)】
  add_noise       : x_σ = (1-σ)·动作 + σ·噪声       ← 训练:给动作掺噪声(对应 x_t)
  training_target : target = 噪声 - 动作              ← 训练:标准方向 u_t
  step            : x_next = x + v·(σ_next - σ)       ← 推理:欧拉法去噪一步(σ_next<σ)
  training_weight : 中间 σ 权重更大                    ← 训练:中间噪声水平更难,多练

【和 modeling_lingbot_vla_v2 的关系】
  FlowMatchingV2 用的是简化版(t 线性 0→1);本调度器是带 shift 的完整版,
  本质公式一致(σ 对应 t)。通常被 pi0 / FlowMatchingV1 路径使用。
====================================================================
"""
import torch


class FlowMatchScheduler:
    """Flow Matching 噪声调度器:管 sigma 时间表 + 加噪/去噪/目标 公式。"""
    def __init__(
        self,
        num_inference_steps=100,        # 推理去噪步数
        num_train_timesteps=1000,       # 训练时间步数(仅用于把 sigma 放大成 timestep)
        shift=3.0,                       # shift 调度:让更多步花在中间噪声水平(类似 SD3)
        sigma_max=1.0,                   # 最大噪声(纯噪声)
        sigma_min=0.003 / 1.002,         # 最小噪声(接近纯动作)
        inverse_timesteps=False,         # 是否反转时间步方向
        extra_one_step=False,            # 是否多算一步
        reverse_sigmas=False,            # 是否反转 sigma
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        """生成 sigma 时间表:从 sigma_max 线性到 sigma_min,再做 shift 变换。"""
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        # ★ shift 变换:σ' = shift·σ / (1 + (shift-1)·σ),把更多步骤分配给中间噪声水平
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps   # 放大成 timestep(便于索引)
        if training:
            # 训练权重:中间 timestep 权重大(中间噪声水平最难学,类似 bsmntw 加权)
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """★ 推理去噪一步(欧拉法):x_next = x + v·(σ_next - σ)。
        因为 σ_next < σ(噪声递减),所以沿 v 走会让样本从噪声走向动作。"""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())   # 找到当前 timestep 对应索引
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0   # 终点 σ=0(纯动作)
        else:
            sigma_ = self.sigmas[timestep_id + 1]                        # 下一个 σ(更小)
        prev_sample = sample + model_output * (sigma_ - sigma)           # 欧拉积分一步
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        """反推:从稳定样本反算 model_output(调试/特殊用途)。"""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep, micro_batch_size, enable_mixed_precision):
        """★ 训练加噪:x_σ = (1-σ)·动作 + σ·噪声。
        对应之前讲的 x_t = t·噪声 + (1-t)·动作(这里 σ 扮演 t 的角色)。"""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        """★ 训练目标(标准方向):target = 噪声 - 动作。
        对应之前讲的 u_t = noise - actions。模型学着预测这个方向。"""
        target = noise - sample
        return target

    def training_weight(self, timestep, micro_batch_size):
        """训练时各时间步的权重:中间噪声水平权重更大(更难学,多练)。"""
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
