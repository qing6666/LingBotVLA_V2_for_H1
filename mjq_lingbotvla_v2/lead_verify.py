#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""取证回放数据核对:chunk_000[0] 对 home state 的逐组超前(不跑模型)"""
import numpy as np
ev = np.loadtxt("/tmp/debug_actions/action_state.csv", delimiter=",", skiprows=1)
S = ev[:, 1:21]          # state 20 维 [腰2,L7,R7,头2,爪2]
A = ev[:, 21:41]         # 执行的动作 20 维
ch0 = np.loadtxt("/tmp/debug_actions/chunk_000.csv", delimiter=",", skiprows=1)  # (50,18)
home = S[0]
print("home state =", np.round(home, 2))
r0 = ch0[0]  # 第一条命令 = 执行帧 f0 的动作(验证:应≈A[0][2:20])
print("\n布局验证: chunk_000[0] vs 执行动作A[0][2:20] 最大差 =", float(np.abs(r0 - A[0][2:20]).max()))
lead = np.abs(r0[0:14] - home[2:16])
print(f"\n[取证回合 第一块首条命令] 超前 home:")
print(f"  左臂 L7 max={lead[:7].max():.3f} rad  逐维 {np.round(lead[:7],2)}")
print(f"  右臂 R7 max={lead[7:14].max():.3f} rad  逐维 {np.round(lead[7:14],2)}")
print(f"  头 {np.round(np.abs(r0[14:16]-home[16:18]),3)}  爪 命令{np.round(r0[16:18],2)} vs state{np.round(home[18:20],2)}")
# 演示对照:数采里动作超前状态的典型值
import pandas as pd
from pathlib import Path
pq = sorted(Path("/home/mjq/robot_item/lingbot-vla-v2-main/H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed/data/chunk-000").glob("file-*.parquet"))[0]
df = pd.read_parquet(pq, columns=["observation.state","action","episode_index","frame_index"])
ep0 = df[df.episode_index==df.episode_index.iloc[0]].sort_values("frame_index")
D_s = np.stack(ep0["observation.state"].values); D_a = np.stack(ep0["action"].values)
dl = np.abs(D_a[:,2:16] - D_s[:,2:16])
print(f"\n[演示对照] |action-state| 臂部:中位 {np.median(dl):.4f} 最大 {dl.max():.3f} rad")
print(f"[演示 f0] 动作 vs 状态 臂部最大差 {np.abs(D_a[0,2:16]-D_s[0,2:16]).max():.4f} rad(开局静止)")
# 执行物理:f0 命令超前 2rad 的话,50 帧内 state 实际跟上多少
print(f"\n[执行追踪] f0→f49 右臂 state 走了 {np.abs(S[49,9:16]-S[0,9:16]).max():.3f} rad / 命令超前 {lead[7:14].max():.3f} rad")
