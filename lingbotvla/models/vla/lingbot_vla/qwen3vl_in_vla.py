# -*- coding: utf-8 -*-
"""
qwen3vl_in_vla.py —— Qwen3-VL 的 LingBot 改装版(monkey patch)
====================================================================
【背景】Qwen3-VL 是官方的视觉语言模型(VLM)。但官方实现不能直接用在 LingBot-VLA 的
       分布式训练框架里(不支持 flex attention / FSDP2 / 序列并行 / 固定 grid 缓存)。

【做法】本文件重写了 Qwen3-VL 的几个关键组件(视觉注意力、视觉块、文本解码层…),
       再用 apply_lingbot_qwen3_vl_patch() 在运行时把官方组件**替换**成定制版(monkey patch)。

【改装了什么】
  Qwen3VLVisionAttention   视觉注意力:用 cu_seqlens 分段做注意力(变长图像 token 高效)
  Qwen3VLVisionBlock       视觉块:用上面的注意力
  Qwen3VLTextDecoderLayer  文本解码层:支持 compute_kqv/output_atten 两步分离(序列并行用)
  Qwen3VLTextModel/Model/ForConditionalGeneration  用定制组件重建的 VLM 主体
  preprcess_grid_thw       预计算图像 patch 网格的位置编码/cu_seqlens(固定分辨率可缓存)
  forward_without_grid_thw 视觉前向:patch_embed → 多层视觉块 → merger + deepstack 特征

【patch 入口】apply_lingbot_qwen3_vl_patch() —— 把上面这些替换进官方模块
【被谁调】modeling_lingbot_vla_v2.py / deploy 都会触发这个 patch
====================================================================
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
import transformers.models.qwen3_vl.modeling_qwen3_vl as hf_qwen3vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
    Qwen3VLModel as _Qwen3VLModel,
    Qwen3VLTextModel as _Qwen3VLTextModel,
    Qwen3VLPreTrainedModel as _Qwen3VLPreTrainedModel,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Qwen3VLVisionMLP,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


def _qwen3vl_no_init_weights(self, module):
    """关闭官方默认的权重初始化(改用从 checkpoint 加载)。"""
    return

_Qwen3VLPreTrainedModel._init_weights = _qwen3vl_no_init_weights   # 替换官方的初始化方法
Qwen3VLPreTrainedModel = _Qwen3VLPreTrainedModel


class Qwen3VLVisionAttention(nn.Module):
    """★ 定制视觉注意力:用 cu_seqlens 把不同图像的 token 分段,各自做注意力(变长高效)。
    支持 flash_attention_2 / sdpa / eager 等多种后端。"""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)   # Q/K/V 一起投影
        self.proj = nn.Linear(self.dim, self.dim)                  # 输出投影
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False                                     # 视觉注意力非因果(双向)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,                 # 累积序列长度,用于分段(每段一张图的 token)
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # Q/K/V 投影并 reshape 成 (head, seq, head_dim)
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)   # 视觉旋转位置编码

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # 按配置选注意力后端(eager / flash_attention_2 / sdpa ...)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # flash_attention_2:用 cu_seqlens 直接算变长(最高效)
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            out_fp32_atten = False
            if key_states.dtype == torch.float32:
                out_fp32_atten = True
                query_states = query_states.to(torch.bfloat16)   # flash 要 bf16
                key_states = key_states.to(torch.bfloat16)
                value_states = value_states.to(torch.bfloat16)
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
            if out_fp32_atten:
                attn_output = attn_output.to(torch.float32)
        else:
            # 非 flash:按 cu_seqlens 把序列切成多段,逐段算注意力再拼回
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    """视觉编码器的一层:LayerNorm → 视觉注意力 → 残差 → LayerNorm → MLP → 残差。"""
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)   # 用定制的视觉注意力
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    """★ 定制文本解码层:支持把注意力拆成「算 qkv」和「算 output」两步(给序列并行/Ulysses 用)。
    三种模式:
      · compute_kqv=True :只算 q/k/v 并返回(序列并行时 qkv 可跨卡切分)
      · output_atten=True:给定 att_output,只做 o_proj + MLP(配合上一步)
      · 都为 False(且给 position_embeddings):走标准完整前向。"""
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: Optional[torch.Tensor] = None,
        start: Optional[int] = 0,
        end: Optional[int] = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if att_output is not None:
            att_output = att_output.to(param_dtype)

        if compute_kqv:
            # 模式①:只算 q/k/v 返回(供序列并行切分)
            hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)
            query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
            key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
            return query_state, key_state, value_state

        if output_atten:
            # 模式②:给定注意力输出 att_output,做 o_proj + MLP
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])
            out_emb += hidden_states
            after_first_residual = out_emb.clone()
            out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            out_emb += after_first_residual
            return out_emb

        position_embeddings = kwargs.pop("position_embeddings", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if position_embeddings is not None:
            # 模式③:标准完整前向(self_attn 内部一次算完)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return residual + hidden_states

        raise ValueError(
            f"Invalid operation compute_kqv={compute_kqv} and output_atten={output_atten} "
            "with Qwen3VLTextDecoderLayer in LingBot-VLA"
        )


class Qwen3VLTextModel(_Qwen3VLTextModel):
    """语言模型主体:embed_tokens + 多层 Qwen3VLTextDecoderLayer + norm + rotary。"""
    def __init__(self, config: Qwen3VLTextConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3VLModel(_Qwen3VLModel):
    """VLM 主体 = 视觉编码器(visual)+ 语言模型(language_model)。"""
    def __init__(self, config: Qwen3VLConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)      # 视觉编码器
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)  # 语言模型
        self.rope_deltas = None
        self.post_init()


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration, GenerationMixin):
    """完整 VLM:Qwen3VLModel + lm_head(语言建模头,VLA 通常不用 lm_head)。"""
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]   # FSDP 切分单元

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


@torch.compiler.disable
def preprcess_grid_thw(self, grid_thw: torch.Tensor):
    """★ 预计算图像 patch 网格(grid_thw)的位置编码和 cu_seqlens。
    固定分辨率时可缓存(use_qwen3_fixed_grid_cache),省去重复计算。"""
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len = int(torch.prod(grid_thw, dim=1).sum().item())
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())                # (cos, sin) 旋转位置编码

    # cu_seqlens:累积序列长度,把不同图像的 token 分段
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return None, position_embeddings, cu_seqlens, split_sizes, max_seqlen


def forward_without_grid_thw(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor = None,
    pos_embeds: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """★ 视觉编码器前向(定制版):patch_embed → 多层视觉块 → merger,并抽出 deepstack 中间特征。
    支持外部预计算好 position_embeddings/cu_seqlens 直接传入(省去重复计算)。"""
    hidden_states = self.patch_embed(hidden_states)   # 图像切块嵌入

    # 若没预计算,现场算
    if pos_embeds is None or position_embeddings is None or cu_seqlens is None or max_seqlen is None:
        pos_embeds, position_embeddings, cu_seqlens, _, max_seqlen = self.preprcess_grid_thw(grid_thw)
    if pos_embeds is None:
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    hidden_states = hidden_states + pos_embeds
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        # 在指定中间层抽出 deepstack 特征(给 VLA 的 deepstack 机制用)
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)   # 把视觉 token 合并(merge_size 倍降采样)
    return hidden_states, deepstack_feature_lists


def apply_lingbot_qwen3_vl_patch():
    """★ Patch 入口:把官方 Qwen3-VL 的组件替换成本文件的定制版(monkey patch)。
    调用后,任何 `from transformers...import Qwen3VL*` 拿到的都是 LingBot 定制版。"""
    logger.info_rank0("apply Qwen3-VL Lingbot patch")
    hf_qwen3vl.Qwen3VLPreTrainedModel = Qwen3VLPreTrainedModel
    hf_qwen3vl.Qwen3VLTextDecoderLayer = Qwen3VLTextDecoderLayer
    hf_qwen3vl.Qwen3VLTextModel = Qwen3VLTextModel
    hf_qwen3vl.Qwen3VLModel = Qwen3VLModel
    hf_qwen3vl.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    hf_qwen3vl.Qwen3VLVisionAttention = Qwen3VLVisionAttention
    hf_qwen3vl.Qwen3VLVisionBlock = Qwen3VLVisionBlock
    hf_qwen3vl.Qwen3VLVisionModel.forward = forward_without_grid_thw
    hf_qwen3vl.Qwen3VLVisionModel.preprcess_grid_thw = preprcess_grid_thw
