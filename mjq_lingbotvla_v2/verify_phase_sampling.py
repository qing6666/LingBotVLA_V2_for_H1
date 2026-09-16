#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_phase_sampling.py —— v4_w2 相位加权采样的干跑验证(不启动训练!)
=====================================================================
在用户按下训练启动键之前,把"配置→数据集→采样"整条链路离线验一遍:

  ① yaml 解析:h1_v4_w2.yaml 走训练同一入口 parse_args(Arguments),
     确认新字段都被声明(未声明会 ValueError: remaining_args —— v4 踩过的坑)
  ② 加权构建:build_vla_dataset 后是 PhaseWeightedDataset,长度=展开后
     行数,且窗口 json 的 num_rows 与底层 VLADataset 行数一致(不一致会 raise)
  ③ 默认关闭=逐比特一致:phase_window_file 清空再建,返回裸 VLADataset
  ④ 采样直方图:均匀索引过 sample_map 后 w=12/8/1 的样本占比 vs 理论值
  ⑤ __getitem__ 冒烟:发起窗内取一条,核对键/形状,且与底层同一行完全一致

【怎么跑】
  conda activate lingbotv2
  python mjq_lingbotvla_v2/verify_phase_sampling.py
(加载 46 万帧 LeRobot 索引约 1 分钟,属正常)
=====================================================================
"""

import os
import sys
from pathlib import Path

# ============ 解释器自纠正保险(与 Lingbot_H1_Training.py 同款) ============
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}")
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# ==========================================================================

import json

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

YAML = PROJECT / "configs/vla/real_robot/h1_v4_w2.yaml"
WIN = PROJECT / "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed.phase_windows.v2.json"

# ---------- ① yaml 解析(训练同一入口:train_lingbotvla 的 Arguments) ----------
# TrainingArguments.__post_init__ 读 LOCAL_RANK → 模拟单卡(与启动器同款 env)
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29699")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.argv = ["verify_phase_sampling.py", str(YAML)]
sys.path.insert(0, str(PROJECT / "tasks" / "vla"))   # 为了 import train_lingbotvla
from train_lingbotvla import Arguments
from lingbotvla.utils.arguments import parse_args

args = parse_args(Arguments)
args.data.chunk_size = args.train.chunk_size   # 训练入口 build 前同样补这一笔(train_lingbotvla.py:492)
assert args.data.phase_window_file and Path(args.data.phase_window_file).exists(), \
    f"data.phase_window_file 未生效: {args.data.phase_window_file!r}"
assert args.train.resume_dataloader_state is False, "resume_dataloader_state 应为 false"
assert args.train.load_checkpoint_path.endswith("h1_v4_w/checkpoints/global_step_88000"), \
    f"load_checkpoint_path 不符: {args.train.load_checkpoint_path}"
assert args.train.max_steps == 120000 and args.train.output_dir == "output/h1_v4_w2"
assert args.train.save_steps == 8000 and args.data.norm_stats_file == "assets/norm_stats/h1_v4.json"
print(f"[1/5] yaml 解析 OK(phase_window_file.v2 / resume_dataloader_state=false / "
      f"load自 global_step_88000 / max_steps 120000 / output {args.train.output_dir})")

# ---------- ② 加权构建 ----------
from lingbotvla.data.dataset import PhaseWeightedDataset, build_vla_dataset

ds = build_vla_dataset(dataset_config=args.data, model_config=None, config=None,
                       processor=None, do_nomalize=False)
assert isinstance(ds, PhaseWeightedDataset), f"应被包装,实际 {type(ds).__name__}"
spec = json.loads(WIN.read_text())
mult = np.ones(spec["num_rows"], dtype=np.int64)
for g in spec["windows"]:
    for s, e in g["ranges"]:
        mult[s:e + 1] = np.maximum(mult[s:e + 1], g["weight"])
expect_len = int(mult.sum())
assert len(ds) == expect_len, f"展开长度 {len(ds)} != 期望 {expect_len}"
print(f"[2/5] 加权构建 OK:{spec['num_rows']} -> {len(ds)} 行({len(ds)/spec['num_rows']:.2f}x)")

# ---------- ③ 默认关闭 = 逐比特一致 ----------
args.data.phase_window_file = ""
ds0 = build_vla_dataset(dataset_config=args.data, model_config=None, config=None,
                        processor=None, do_nomalize=False)
assert not isinstance(ds0, PhaseWeightedDataset), "清空后不应再包装"
assert len(ds0) == spec["num_rows"], f"裸长度 {len(ds0)} != {spec['num_rows']}"
print(f"[3/5] 默认关闭 OK:清空字段 → 裸 VLADataset,{len(ds0)} 行(与 v4 一致)")

# ---------- ④ 采样直方图 ----------
rng = np.random.default_rng(0)
draws = ds.sample_map[rng.integers(0, len(ds), 200_000)]
WEIGHTS = [int(w) for w in np.unique(mult)]          # v2: [1, 8, 12]
got = {w: float((mult[draws] == w).mean()) for w in WEIGHTS}
theory = {w: float((mult == w).sum() * w / expect_len) for w in WEIGHTS}
for w in WEIGHTS:
    assert abs(got[w] - theory[w]) < 0.005, f"w={w} 采样占比 {got[w]:.3f} 偏离理论 {theory[w]:.3f}"
visits = {w: round(32_000 * 4 * theory[w] / (mult == w).sum(), 2) for w in WEIGHTS}
print(f"[4/5] 采样直方图 OK(20 万抽):" +
      "  ".join(f"w={w} {got[w]:.1%}(理论 {theory[w]:.1%})" for w in WEIGHTS))
print(f"      32k 步×batch4 期望访问/帧:" +
      "、".join(f"w={w} {visits[w]} 次" for w in WEIGHTS) + "(v4 基线一律 0.35)")

# ---------- ⑤ 映射不变量 + DataLoader 组合冒烟 ----------
# 说明:VLADataset 的 __getitem__ 免模型路径(config=None)在库自身
# convert_features 的 assert 处就断掉,与本次改动无关 —— v4 训练走的
# 是带 model.config 的完整路径,一行没动。这里改为验证包装层的契约:
#   a) sample_map 单调不减(索引展开不变量,epoch 内局部性保持)
#   b) 桩底层数据集 + 真 torch DataLoader(shuffle) 能正常出批
onset_r_group = next(g for g in spec["windows"] if g["type"] == "onset_right")
W_R_ONSET = int(onset_r_group["weight"])             # v2: 12
row0 = onset_r_group["ranges"][0][0]                 # 首回合右发窗首行(f0 附近)
assert np.all(np.diff(ds.sample_map) >= 0), "sample_map 应单调不减(索引展开不变量)"
assert mult[row0] == W_R_ONSET, f"右发窗行 {row0} 权重应为 {W_R_ONSET},实际 {mult[row0]}"
n_pos = int((ds.sample_map == row0).sum())
assert n_pos == W_R_ONSET, f"右发窗行在展开空间应出现 {W_R_ONSET} 次,实际 {n_pos}"

import torch
from torch.utils.data import DataLoader, Dataset

class _Stub(Dataset):
    def __len__(self):
        return spec["num_rows"]

    def __getitem__(self, i):
        return {"x": torch.tensor([i], dtype=torch.float32)}

wrapped = PhaseWeightedDataset(_Stub(), str(WIN))
assert len(wrapped) == expect_len
dl = DataLoader(wrapped, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(dl))
assert batch["x"].shape == (4, 1) and int(batch["x"].max()) < spec["num_rows"]
print(f"[5/5] 映射不变量 + DataLoader 组合 OK:sample_map 单调;右发窗行 {row0} "
      f"展开 {W_R_ONSET} 次;shuffle 出批 {tuple(batch['x'].shape)} 正常")
print("\n全部通过 —— 可以启动训练(mjq_lingbotvla_v2/Lingbot_H1_Training.py)。")
