#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_phase_windows.py —— 生成"相位加权采样"的窗口文件(方案 A/A2 配套工具)
=====================================================================
【背景】v4 闭环 0/10 的主根因③:模型没学会动作"发起段"和"终段逼近/闭爪"
(40k 步仅 0.35 epoch,发起帧平均被采样 0.35 次;详见项目根《V4闭环失败诊断.md》)。
对策:把这两类相位的帧在采样空间里复制多份(索引展开),再续训。
v1(发起×8/抓取×4)续训到 88000:发起段证实有效(左 52%/右方向对),但
①右发起窗 [onset−15,+75] 恰好把"双臂全静止"的 f0-10 切在窗外;
②抓取 ×4 均匀一大片把"停住别闭"稀释成"接近=满闭"(GR 真值渐闭 −0.42,
预测满闭 −1.02)。v2 依此调窗口/权重(见下 SCHEME 注释,依据为 FK 实测:
操作员停顿点=爪心距方块 5cm,模型停 6-9cm,差 1.5-4cm≈10-20px,可分辨)。

【这个脚本做什么】
  扫描 v4 裁剪副本的全部 parquet,逐回合定位 4 个事件:
    onset_right  : 右臂首次持续动起来(串行协议 302/302 右臂先动,中位 f24)
    onset_left   : 左臂首次持续动起来(中位 f766,右臂完工后)
    grasp_right  : 右爪开始闭合(action[19] 首次 < 0.5;中位 f313)
    grasp_left   : 左爪开始闭合(action[18] 首次 < 0.5)
  以事件为锚开窗(帧号相对回合内,再换算成全数据集绝对行号),窗与权重见 SCHEME:
    发起窗 [onset−30, onset+75]  右×12 / 左×8 —— 前移 30 覆盖 f0-10 的全静止帧
    抓取窗 [grasp−60, grasp+30]  ×8 集中 —— 只放大"最后逼近平移+停顿+起闭",
                                          更早的逼近段降回 ×1(v1 是 [−100,+80]×4)
  输出 json 写在数据集旁边(不进数据集目录,不动数据集本身):
    <数据集>.phase_windows.v2.json

【20 维动作下标】[0:2]腰 [2:9]左臂 [9:16]右臂 [16:18]头(yaw,pitch) [18]左爪 [19]右爪
  (夹爪 open=1.0 / closed≈0.21)

【怎么跑】
  conda activate lingbotv2
  python mjq_lingbotvla_v2/make_phase_windows.py            # 用上面 SCHEME 默认值
【怎么验证】训练侧加载见 lingbotvla/data/dataset.py 的 PhaseWeightedDataset;
  干跑验证脚本 mjq_lingbotvla_v2/verify_phase_sampling.py。
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

import argparse
import json

import numpy as np
import pandas as pd

# ==================== 配置(改成你的参数)====================
DATASET = "/home/mjq/robot_item/lingbot-vla-v2-main/H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"
N_PARQUET = 8          # data/chunk-000/file-000..007.parquet
# =============================================================

# ---- v2 相位窗方案(2026-08-31,依据 v4_w 88000 闭环+教师强制+FK 实测,见
#      《V4闭环失败诊断.md》七/八章)----
#   ① 右发起 ×12 且窗前移 −30(覆盖 f0-10:v1 的 onset−15 恰好把 rollout 起始
#      最像的"双臂全静止"帧切在窗外;88000 死锁 4/10);
#   ② 左发起 ×8 同窗(已学到 52%,维持压力收满);
#   ③ 抓取集中 [−60,+30] ×8(v1 均匀 ×4 一大片把"停住别闭"样本稀释成了
#      "逼近=满闭";FK 实测操作员停顿点=爪心距方块 5cm,模型停 6-9cm,
#      差的就是最后这段逼近平移+停顿+起闭);更早逼近段降回 1。
SCHEME = {
    #            前扩   后扩   权重
    "onset_right": (30, 75, 12),
    "onset_left":  (30, 75, 8),
    "grasp_right": (60, 30, 8),
    "grasp_left":  (60, 30, 8),
}
# 期望访问次数按"88000→120000 续 32000 步 × batch4"打印
STEPS_FOR_VISITS = 32_000

# 动作维度(20 维数据集口径)
D_L_ARM, D_R_ARM, D_GRIP_L, D_GRIP_R = 2, 9, 18, 19
# 事件判据(与 v1 相同)
ONSET_DEV_THR = 0.15      # 单帧偏移阈值(rad;七维取最大)
ONSET_SUSTAIN_N = 30      # 持续确认窗(帧):其后 30 帧平均偏移也要超阈值
ONSET_SUSTAIN_THR = 0.20
GRIP_CLOSE_THR = 0.5      # 夹爪低于此值 = 开始闭合


def find_onset(dev: np.ndarray) -> int:
    """首次'持续动起来'的帧:单帧超阈值且其后 30 帧均值也超。找不到返回 -1。"""
    T = len(dev)
    for t in range(T):
        if dev[t] > ONSET_DEV_THR:
            if dev[t:min(t + ONSET_SUSTAIN_N, T)].mean() > ONSET_SUSTAIN_THR:
                return t
    return -1


def main() -> None:
    ap = argparse.ArgumentParser(description="生成相位加权采样窗口 json(v2 方案)")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--n-parquet", type=int, default=N_PARQUET)
    ap.add_argument("--out", default=None, help="默认写到 <dataset>.phase_windows.v2.json")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    files = sorted(ds_path.glob("data/chunk-000/*.parquet"))
    assert len(files) >= args.n_parquet, f"parquet 数量不符:找到 {len(files)}"
    files = files[: args.n_parquet]

    # 与训练侧完全相同的读取顺序(文件名排序 concat)→ 绝对行号才对得上
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    num_rows = len(df)
    actions = np.stack(df["action"].apply(np.asarray))          # (num_rows, 20)
    ep_ids = df["episode_index"].to_numpy()
    fidx = df["frame_index"].to_numpy()

    groups = df.groupby("episode_index", sort=False)
    n_ep = len(groups)
    win = {
        "onset_right": [], "onset_left": [],
        "grasp_right": [], "grasp_left": [],
    }
    stats = {"onset_right": [], "onset_left": [], "grasp_right": [], "grasp_left": []}
    ep_rows_list = []
    for ep, g in groups:
        rows = g.index.to_numpy()          # 该回合的绝对行号(应连续)
        ep_rows_list.append((rows[0], rows[-1]))
        A = actions[rows]                   # (T, 20) 回合内动作
        T = len(A)
        # 发起检测:七维关节相对 f0 的最大偏移
        dev_r = np.abs(A[:, D_R_ARM:D_R_ARM + 7] - A[0, D_R_ARM:D_R_ARM + 7]).max(axis=1)
        dev_l = np.abs(A[:, D_L_ARM:D_L_ARM + 7] - A[0, D_L_ARM:D_L_ARM + 7]).max(axis=1)
        anchors = {
            "onset_right": find_onset(dev_r),
            "onset_left": find_onset(dev_l),
            "grasp_right": int(np.argmax(A[:, D_GRIP_R] < GRIP_CLOSE_THR)) if (A[:, D_GRIP_R] < GRIP_CLOSE_THR).any() else -1,
            "grasp_left": int(np.argmax(A[:, D_GRIP_L] < GRIP_CLOSE_THR)) if (A[:, D_GRIP_L] < GRIP_CLOSE_THR).any() else -1,
        }
        for name, a in anchors.items():
            if a < 0:
                continue
            pre, post, w = SCHEME[name]
            stats[name].append(int(fidx[rows[a]]))    # 回合内帧号(可读性)
            s, e = max(0, a - pre), min(T - 1, a + post)
            win[name].append([int(rows[s]), int(rows[e])])   # 绝对行号,闭区间

    # ---- 汇总统计(权重预演,与训练侧同一套逻辑) ----
    mult = np.ones(num_rows, dtype=np.int64)
    for name in win:
        _, _, w = SCHEME[name]
        for s, e in win[name]:
            mult[s:e + 1] = np.maximum(mult[s:e + 1], w)      # 重叠取最大,不叠乘
    map_len = int(mult.sum())
    for name in ("onset_right", "onset_left", "grasp_right", "grasp_left"):
        pre, post, w = SCHEME[name]
        cov = sum(e - s + 1 for s, e in win[name]) / num_rows
        v = stats[name]
        v_med = float(np.median(v)) if v else float("nan")
        print(f"{name:12s} x{w} 窗[{-pre},{post}] 回合 {len(win[name]):3d}/{n_ep} "
              f"锚点帧中位 {v_med:7.1f} 覆盖 {cov:5.1%}")
    share = {w: float((mult == w).mean()) for w in np.unique(mult)}
    print(f"\n权重分布(帧占比): " + "  ".join(f"w={w}: {sh:.1%}" for w, sh in share.items()))
    print(f"索引展开后总长度: {num_rows} -> {map_len} ({map_len / num_rows:.2f}x)")
    # 续训步 × batch4 的样本量下,各类帧的期望访问次数(v4 基线一律 0.35):
    for w in sorted(share):
        n_w = int((mult == w).sum())
        visits = STEPS_FOR_VISITS * 4 * (n_w * w) / map_len / n_w
        print(f"  w={w} 帧 {n_w} 个: 期望访问 {visits:.2f} 次/帧")

    out = {
        "dataset": str(ds_path),
        "num_rows": num_rows,
        "num_episodes": n_ep,
        "note": "v2 方案(见 make_phase_windows.py SCHEME 注释);ranges 为全数据集绝对行号"
                "(闭区间);weight=采样空间复制倍数,重叠取最大",
        "windows": [
            {"type": k, "weight": SCHEME[k][2], "ranges": v} for k, v in win.items()
        ],
    }
    out_path = Path(args.out) if args.out else ds_path.parent / (ds_path.name + ".phase_windows.v2.json")
    out_path.write_text(json.dumps(out))
    print(f"\n已写出: {out_path}")


if __name__ == "__main__":
    main()
