# Copyright 2026 Robbyant Team and/or its affiliates
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
configuration_lingbot_vla.py —— LingBot-VLA 模型配置(设计图纸)
====================================================================
【作用】定义模型的所有结构超参数(动作维度、MoE 专家数、去噪步数、蒸馏配置…)。
       它是模型的"设计图纸":改这里 = 改模型结构。

【配置怎么来】
  yaml(configs/vla/.../*.yaml)的 model/train 段
     → train_lingbotvla.py 里 config_kwargs = {**vars(args.model), **vars(args.train)}
     → config_cls(**config_kwargs)  建出本配置对象
     → build_foundation_model 按 config 建模型

【两个类】
  LingbotVLAConfig   V1 基类配置(通用参数)
  LingbotVLAV2Config V2 配置(继承 V1,设 Qwen3-VL 默认值:flex_cached 注意力等)
  LingBot-VLA 2.0 用 LingbotVLAV2Config(对应 yaml 里 config_key: LingbotVLAV2Config)。
====================================================================
"""

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

from transformers import AutoConfig, PretrainedConfig

class LingbotVLAConfig(PretrainedConfig):
    """Configuration class for Lingbot-VLA.
    This is the configuration class to store the configuration of a [`Lingbot-VLA`].
    LingBot-VLA V1 的配置(也是 V2 的基类)。
    """

    model_type = "lingbotvla"
    is_composition = True

    def __init__(
        self,
        # ---- 路径类:模型/视觉/分词器位置 ----
        vlm_repo_id: Optional[str] = None,
        expert_vision_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,

        # ---- 训练模式 & AdaNorm(时间嵌入)----
        post_training: bool = False,            # 是否后训练(微调)
        adanorm_time: bool = False,             # action expert 的 AdaNorm 是否加时间嵌入
        split_gate_liner: bool = False,
        nosplit_gate_liner: bool = False,
        separate_time_proj: bool = False,
        final_norm_adanorm: bool = False,       # 最终 norm 是否用 AdaNorm

        # ---- 视觉编码器 ----
        enable_expert_vision: bool = False,
        expert_vision_type: Optional[str] = None,
        freeze_vision_encoder: bool = False,    # 是否冻结视觉编码器

        # ---- 增量训练 ----
        incremental_training: bool = False,
        depth_incremental_training: bool = False,
        reinit_mismatched_weights: bool = False, # 结构不匹配的权重是否重新初始化

        # ---- 动作空间 & Flow Matching ----
        action_dim: int = 14,                   # 实际动作维度
        max_action_dim: int = 14,               # 动作 padding 后的维度(统一空间)
        max_state_dim: int = 14,                # 状态 padding 后的维度
        chunk_size: int = 50,                   # 动作序列长度(一次预测多少步)
        vlm_causal: bool = False,               # VLM 图像/语言 token 是否用因果注意力
        tokenizer_max_length: int = 48,         # 语言指令最大 token 长度
        loss_type: str = "fm",                  # loss 类型:"fm"(MSE flow matching)/ "L1_fm"
        action_loss_weights: Optional[List[float]] = None,   # 统一动作空间逐维损失权重(长度=max_action_dim;
                                                              #   None=不加权,行为与历史完全一致)
        norm_qkv: bool = False,
        align_params: Optional[Dict[str, Any]] = None,   # ★ 蒸馏配置(depth/video 教师参数)
        use_compile: bool = False,              # 是否 torch.compile

        # ---- MoE(混合专家)----
        use_moe: bool = False,                  # 是否启用 MoE
        token_moe_layers: Optional[list] = None, # 哪些层用 MoE(如 [0..35])
        token_num_experts: int = 32,            # ★ 专家数(32)
        token_top_k: int = 1,                   # 每个 token 选几个专家(V2 实际用 4)
        token_moe_intermediate_size: int = 256, # 路由专家 FFN 中间维度
        token_shared_intermediate_size: int = 256, # 共享专家 FFN 中间维度
        bias_update_speed: float = 0.001,       # ★ loss-free 均衡 bias 更新速度
        sequence_wise_loss_coeff: float = 0.001,# 序列级均衡 loss 系数
        sequence_wise_mode: str = "per_sequence",
        router_z_loss_coeff: float = 0.0,       # router z-loss 系数(数值稳定)
        router_activation: str = "softmax",     # 路由器激活:"softmax"/"sigmoid"
        routed_scaling_factor: float = 1.0,     # 路由权重缩放因子
        use_shared_expert_gate: bool = True,    # 共享专家是否加 gate
        moe_implementation: Optional[Literal[None, "eager", "fused"]] = None,  # MoE 实现(V2 用 fused)
        split_fused_experts_from_decoder_fsdp: bool = False,

        # ---- action expert 结构 ----
        expert_hidden_size: int = 768,          # 动作专家隐藏维度
        expert_intermediate_size: int = 2752,   # 动作专家 FFN 中间维度
        action_num_attention_heads: int = 16,   # 动作专家注意力头数
        action_num_key_value_heads: int = 2,    # KV 头数(GQA)
        action_head_dim: int = 128,             # 每个头的维度
        action_fp32: bool = False,              # 动作/状态是否用 fp32

        # ---- Qwen3-VL 专用 ----
        use_qwen3_chat_template: bool = False,  # 用 Qwen3 chat template 组装指令
        return_image_grid_thw: bool = False,    # 返回图像 patch 网格(位置编码用)
        qwen3vl_use_vision_boundaries: bool = False,  # 图像 token 加边界标记
        precompute_grid_thw: bool = False,      # 预计算缓存 grid_thw
        use_qwen3_fixed_grid_cache: bool = False,

        # ---- VLM 输出 & 注意力 ----
        use_lm_head: bool = False,              # 是否用语言模型头(VLA 通常不用)
        vocab_size: int = 0,                    # 词表大小(0 则按 vlm 自动推断)
        vit_attn_implementation: str = "flash_attention_2",  # 视觉塔注意力实现
        attention_implementation: str = "flex",  # VLA 注意力实现(flex/flex_cached/eager)

        # ---- 训练范围 ----
        train_expert_only: bool = False,        # 只训动作专家
        train_state_proj: bool = True,          # 是否训状态投影

        **kwargs
    ):
        super().__init__()
        if moe_implementation is None:
            moe_implementation = kwargs.pop("_moe_implementation", None)
        self.architectures = ["LingbotVlaPolicy"]
        self.train_state_proj = train_state_proj
        self.train_expert_only = train_expert_only
        self.use_cache = False
        self.attention_implementation = attention_implementation
        self.num_steps = 10                     # ★ 推理去噪步数(flow matching 积分步数)
        self.n_obs_steps = 1                    # 观测步数

        assert not (split_gate_liner and nosplit_gate_liner), \
            "split_gate_liner and nosplit_gate_liner can not be both True."

        self.vlm_repo_id = vlm_repo_id
        self.expert_vision_path = expert_vision_path
        self.tokenizer_path = tokenizer_path
        self.post_training = post_training
        self.adanorm_time = adanorm_time
        self.split_gate_liner = split_gate_liner
        self.nosplit_gate_liner = nosplit_gate_liner
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.incremental_training = incremental_training
        self.depth_incremental_training = depth_incremental_training
        self.reinit_mismatched_weights = reinit_mismatched_weights
        self.norm_qkv = norm_qkv
        self.use_compile = use_compile
        self.loss_type = loss_type
        self.action_loss_weights = action_loss_weights
        self.separate_time_proj = separate_time_proj
        self.final_norm_adanorm = final_norm_adanorm
        self.freeze_vision_encoder = freeze_vision_encoder
        self.tokenizer_max_length = tokenizer_max_length
        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.chunk_size = chunk_size
        self.n_action_steps = chunk_size         # n_action_steps = chunk_size(动作序列长度)
        self.vlm_causal = vlm_causal
        self.align_params = align_params
        self.use_moe = use_moe
        if self.use_moe:                          # 只有启用 MoE 才存这些参数
            self.token_moe_layers = token_moe_layers
            self.token_num_experts = token_num_experts
            self.token_top_k = token_top_k
            self.token_moe_intermediate_size = token_moe_intermediate_size
            self.token_shared_intermediate_size = token_shared_intermediate_size
        self.bias_update_speed = bias_update_speed
        self.sequence_wise_loss_coeff = sequence_wise_loss_coeff
        self.sequence_wise_mode = sequence_wise_mode
        self.router_z_loss_coeff = router_z_loss_coeff
        self.router_activation = router_activation
        self.routed_scaling_factor = routed_scaling_factor
        self.use_shared_expert_gate = use_shared_expert_gate
        self.moe_implementation = moe_implementation
        if moe_implementation is not None:
            if moe_implementation not in ("eager", "fused"):
                raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
            self._moe_implementation = moe_implementation
        self.split_fused_experts_from_decoder_fsdp = split_fused_experts_from_decoder_fsdp
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        self.action_fp32 = action_fp32
        self.use_qwen3_chat_template = use_qwen3_chat_template
        self.return_image_grid_thw = return_image_grid_thw
        self.qwen3vl_use_vision_boundaries = qwen3vl_use_vision_boundaries
        self.precompute_grid_thw = precompute_grid_thw
        self.use_qwen3_fixed_grid_cache = use_qwen3_fixed_grid_cache
        self.use_lm_head = use_lm_head
        # vocab_size 未指定时,按 VLM 家族自动推断
        if vocab_size == 0:
            if vlm_repo_id and 'paligemma' in vlm_repo_id.lower():
                self.vocab_size = 257216
            elif vlm_repo_id and 'qwen' in vlm_repo_id.lower():
                self.vocab_size = 151936                        # Qwen 系列词表
            else:
                self.vocab_size = 257152
        else:
            self.vocab_size = vocab_size
        self.vit_attn_implementation = vit_attn_implementation

class LingbotVLAV2Config(LingbotVLAConfig):
    """LingBot-VLA 2.0 配置(继承 V1,设 Qwen3-VL 默认值)。"""
    def __init__(self, **kwargs):
        # V2 专属默认值(用 setdefault:命令行/yaml 显式传的优先)
        kwargs.setdefault("attention_implementation", "flex_cached")  # V2 用 flex_cached 注意力
        kwargs.setdefault("vit_attn_implementation", "flash_attention_2")
        kwargs.setdefault("action_num_attention_heads", 32)           # V2 动作专家更多头
        kwargs.setdefault("action_num_key_value_heads", 8)
        kwargs.setdefault("action_head_dim", 128)
        kwargs.setdefault("expert_hidden_size", 768)
        kwargs.setdefault("use_qwen3_chat_template", True)            # V2 用 Qwen3 chat template
        kwargs.setdefault("return_image_grid_thw", True)
        kwargs.setdefault("qwen3vl_use_vision_boundaries", True)
        kwargs.setdefault("use_qwen3_fixed_grid_cache", True)
        super().__init__(**kwargs)
        self.architectures = ["LingbotVlaV2Policy"]                   # V2 策略类名
        self.vlm_family = "qwen3_vl"                                  # V2 用 Qwen3-VL 家族


ConfigClass = [LingbotVLAConfig, LingbotVLAV2Config]
__all__ = ["LingbotVLAConfig", "LingbotVLAV2Config"]
