#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行器往返一致性审计(按《模型输出与动作执行器排查方案》§十六/§十九)
==========================================================
问题:训练正向(FeatureTransform.apply) 与 推理还原(FeatureTransform.unapply)
     是否互逆?norm/映射/槽位/单位 在训练与推理两端是否一致?

方法:
  A. 动作往返:取数据集真实 (state, action[50帧]) → 走训练正向得到归一化 55 维块
     → 假装模型完美输出了它 → 走 deploy 还原 → 应逐位复原数据集动作(18 维)。
     若恒等 ⇒ norm+映射+槽位+单位 全链路一致(误差只剩 float 精度)。
  B. state 对称:同一 20 维 state 分别走"训练口径 apply"与"推理口径 apply(policy_eval=True)",
     归一化 55 维 state 应完全相同。
  C. 槽位图:归一化动作块 55 维里信号只应落在 L臂[0:7]/R臂[7:14]/夹爪[28:30]/头[34:36],
     其余槽(end/腰/base/hand/尾)应恒 0 —— 与 action_loss_weights 的下标注释互证。
  D. 归一化范围:各槽位 max|归一化值| 应在 ±4 内(检测 std 地板失效/除以极小 std 的爆掉)。
  E. norm 统计文件:训练 yaml 声明的 norm_stats_file 与 rollout 默认应同一份。
  F. 夹爪量纲:数据集夹爪 action/state 应在 [0,1](actuator_length gear 归一化口径)。
"""
import sys
import os
from pathlib import Path

# ---- 解释器自纠正(必须 lingbotv2) ----
_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _PY and Path(_PY).exists():
    os.execv(_PY, [_PY] + sys.argv)

import json
import yaml
import numpy as np
import torch
from types import SimpleNamespace

PROJECT_ROOT = Path("/home/mjq/robot_item/lingbot-vla-v2-main")
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.models import build_processor
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.data.vla_data.utils import FeatureTransform

CKPT = PROJECT_ROOT / "output/h1_v4_w2/checkpoints/global_step_120000/hf_ckpt"
TRAIN_YAML = CKPT.parent.parent.parent / "lingbotvla_cli.yaml"   # deploy 同款相对定位
ROBOT_CFG = "configs/robot_configs/h1.yaml"
NORM_STATS = "assets/norm_stats/h1_v4.json"                      # rollout --norm-path 默认
DATA_ROOT = PROJECT_ROOT / "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"
TASK = "Pick up the red cube and put it into the green bin"

# ============ 0) 仿 deploy load_vla 构建 FeatureTransform(不加载 6B 模型) ============
tc = yaml.safe_load(open(TRAIN_YAML))
training_model_config = dict(tc["model"]); training_model_config.update(tc["train"])
config = LingbotVLAV2Config(**training_model_config)
for k, v in training_model_config.items():
    if not hasattr(config, k):
        setattr(config, k, v)
base_model_path = tc["model"]["tokenizer_path"]
processor = build_processor(base_model_path)
data_config = SimpleNamespace(**tc["data"])
ft = FeatureTransform(ROBOT_CFG, data_config, config, processor,
                      chunk_size=config.chunk_size, norm_stats_path=NORM_STATS)
print(f"[0] FeatureTransform 就绪 chunk={config.chunk_size} | 训练yaml={TRAIN_YAML.name}")

# ============ 1) 读数据集真实样本 ============
import pandas as pd
pq = sorted((DATA_ROOT / "data" / "chunk-000").glob("file-*.parquet"))[0]
df = pd.read_parquet(pq, columns=["observation.state", "action", "episode_index", "frame_index"])
ep0 = df[df.episode_index == df.episode_index.iloc[0]].sort_values("frame_index")
T0 = 200                                                    # 回合中段,动作丰富
st20 = torch.tensor(np.stack(ep0["observation.state"].iloc[T0]), dtype=torch.float32)
act50 = torch.tensor(np.stack(ep0["action"].iloc[T0:T0 + 50]), dtype=torch.float32)
print(f"[1] 样本 ep{ep0.episode_index.iloc[0]} f{T0}..f{T0+49} | state{tuple(st20.shape)} action{tuple(act50.shape)}")
grip_a = act50[:, 18:20].numpy(); grip_s = ep0["observation.state"].apply(lambda v: v[18:20])
grip_s = np.stack(grip_s.values)
print(f"[F] 夹爪量纲 action[{grip_a.min():.3f},{grip_a.max():.3f}] state[{grip_s.min():.3f},{grip_s.max():.3f}] (期望都在0..1)")

def zeros_img():
    return torch.zeros(256, 256, 3, dtype=torch.uint8)

IMG_KEYS = ["observation.images.head_rgb", "observation.images.left_wrist_rgb",
            "observation.images.right_wrist_rgb"]

# ============ 2) A: 动作往返 训练apply → deploy unapply ============
train_item = {k: zeros_img() for k in IMG_KEYS}
train_item.update({"observation.state": st20, "action": act50,
                   "action_is_pad": torch.zeros(50, dtype=torch.bool), "task": TASK})
batch_train = ft.apply(train_item, policy_eval=False)          # 训练口径(带 action)
norm_chunk = batch_train["actions"]                            # (50,55) 归一化+padding

eval_obs = {k: zeros_img() for k in IMG_KEYS}
eval_obs.update({"observation.state": st20, "task": TASK})
applied = ft.apply(eval_obs, policy_eval=True)                 # deploy 口径(不带 action)
restore_in = dict(applied); restore_in["actions"] = norm_chunk # 假装模型完美输出训练目标
recovered = ft.unapply(restore_in)                             # deploy 还原链路

rec = recovered["action"]
rec = rec.numpy() if isinstance(rec, torch.Tensor) else np.asarray(rec)
orig18 = act50[:, 2:20].numpy()                                # 数据集 18 维(腰不映射)
err = np.abs(rec - orig18)
groups = {"L臂[0:7]": (0, 7), "R臂[7:14]": (7, 14), "头[14:16]": (14, 16), "夹爪[16:18]": (16, 18)}
print(f"\n[A] 动作往返:还原形状 {rec.shape} (期望 (50,18) → rollout 会补腰2维成20)")
ok_a = True
for g, (a, b) in groups.items():
    e = err[:, a:b].max()
    ok_a &= e < 1e-4
    print(f"    {g:10s} max|还原-原始| = {e:.3e}")
print(f"[A] >>> {'PASS 恒等(norm+映射+槽位+单位全链路一致)' if ok_a and rec.shape==(50,18) else 'FAIL'}")

# ============ 3) B: state 训练/推理口径对称 ============
s_train = batch_train["state"]; s_eval = applied["state"]
d = (s_train - s_eval).abs().max().item()
print(f"\n[B] state 对称: max|train口径-eval口径| = {d:.3e} → {'PASS' if d < 1e-6 else 'FAIL'}")
print(f"    state 形状 {tuple(s_eval.shape)} | max|norm| = {s_eval.abs().max():.3f} (期望<4)")

# ============ 4) C/D: 槽位图 + 归一化范围 ============
nc = norm_chunk.numpy()
slots = {"L臂[0:7]": (0, 7), "R臂[7:14]": (7, 14), "end[14:28]": (14, 28), "夹爪[28:30]": (28, 30),
         "腰[30:34]": (30, 34), "头[34:36]": (34, 36), "base[36:39]": (36, 39),
         "hand[39:51]": (39, 51), "尾[51:55]": (51, 55)}
print("\n[C/D] 55 维槽位图(归一化动作块):           max|·|     (信号槽应>0,空槽应=0)")
ok_c = True
for g, (a, b) in slots.items():
    m = np.abs(nc[:, a:b]).max()
    expect_sig = g in ("L臂[0:7]", "R臂[7:14]", "夹爪[28:30]", "头[34:36]")
    if expect_sig: ok_c &= m > 0.05
    else:          ok_c &= m == 0.0
    flag = "信号" if m > 0.05 else ("空" if m == 0.0 else "★异常")
    rng = f"范围[{nc[:, a:b].min():+.2f},{nc[:, a:b].max():+.2f}]" if m > 0 else ""
    print(f"    {g:12s} {m:8.4f}  {flag}  {rng}")
print(f"[C] >>> {'PASS 槽位与 action_loss_weights 下标注释一致' if ok_c else 'FAIL'}")

# ============ 5) E: norm 统计文件一致性 ============
yaml_norm = tc["data"]["norm_stats_file"]
same = Path(str(PROJECT_ROOT / yaml_norm) if not os.path.isabs(yaml_norm) else yaml_norm).resolve() == \
       (PROJECT_ROOT / NORM_STATS).resolve()
print(f"\n[E] 训练yaml norm={yaml_norm} | rollout默认={NORM_STATS} → {'PASS 同一份' if same else 'FAIL 不一致!'}")
ns = json.load(open(PROJECT_ROOT / NORM_STATS))
print(f"    统计键: {list(ns.keys())[:6]}")
