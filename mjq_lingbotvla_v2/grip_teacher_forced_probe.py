#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grip_teacher_forced_probe.py —— 开环教师强制闭爪核查
====================================================================
【回答一个问题】闭环里"贴近方块爪不闭"(左)与"远处空中狂闭"(右),
到底是哪一种病:
  A. 闭环状态漂出训练分布:模型在演示观测上其实会闭,闭环观测没触发
     → 查闭环观测域(不用再训练)
  B. 抓取映射没学会:连演示观测都激不发闭合
     → 加权重采/续训/租卡讨论才有依据

【做法】(教师强制 = 完全演示分布,无闭环漂移)
  1. 读 v4 数据集 parquet,定位每回合真实闭爪时刻(该爪 action 首次 <0.5);
  2. 在闭爪时刻前后取若干帧的真实观测(数据集视频解码的图像+state),
     喂模型(与 rollout 同一条 LingbotVLAv2Server 推理路径,chunk_ret=True
     拿整块 50 帧规划);
  3. 看预测动作块里夹爪维(dim18 左爪 / dim19 右爪)闭不闭,与真值块对比。

【判读】anchor(offset=0)处目标爪的预测块内最小值 pred_min:
  <0.5 占多数   → 会闭 → 结论 A(闭环分布漂移)
  ≥0.85         → 不闭 → 结论 B(映射未学会)
  中间/只在 offset>0 闭 → 部分学会或相位滞后,看侧写

【用法】(lingbotv2 环境,或直接跑——有解释器自纠正)
  python mjq_lingbotvla_v2/grip_teacher_forced_probe.py
  # 可选:--episodes 5 30 55 ...(默认 12 回合铺满 0-301)
  #       --offsets -45 -30 -15 0 15 | --csv /tmp/xxx.csv
====================================================================
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

# ============ 解释器自纠正保险(同训练/评估/rollout 启动器) ============
# ★必须放在 import torch/pandas 之前:base 环境没有这些包
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}", flush=True)
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# =============================================================

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))            # deploy.lingbot_vla_v2_policy
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # open_loop_eval(复用观测准备)

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # noqa: E402
from open_loop_eval import prepare_eval_observation, to_numpy  # noqa: E402
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset  # noqa: E402

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    LEROBOT_API = "v3"
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    LEROBOT_API = "v2"

# ---- 数据集口径常量(与 make_phase_windows / grip_data_check 一致)----
GRIP_L, GRIP_R = 18, 19     # action 维:左爪开度 / 右爪开度(1=开,≈0.21=闭)
CLOSE_TH = 0.5              # 判"闭"阈值(同数据侧 close_segs)
MIN_SEG = 5                 # 连续闭合段最短帧数(防毛刺)

DEFAULT_CKPT = PROJECT_ROOT / "output/h1_v4_full/checkpoints/global_step_40000/hf_ckpt"
DEFAULT_DATA = PROJECT_ROOT / "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"
DEFAULT_NORM = PROJECT_ROOT / "assets/norm_stats/h1_v4.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="开环教师强制闭爪核查")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--norm-path", type=str, default=str(DEFAULT_NORM),
                   help="必须与训练同一份统计表")
    p.add_argument("--episodes", type=int, nargs="+",
                   default=[5, 30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280],
                   help="要核查的回合号(默认 12 回合铺满 0-301)")
    p.add_argument("--offsets", type=int, nargs="+", default=[-45, -30, -15, 0, 15],
                   help="相对闭爪时刻的探测帧偏移(帧 @30Hz;0=正好闭爪瞬间)")
    p.add_argument("--csv", type=str, default="/tmp/grip_teacher_forced.csv")
    return p.parse_args()


def episode_grip_onsets(data_root: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """读全数据集 parquet → ({ep: (onset_L, onset_R, 首行号, 帧数)}, 全量 action, 全量 state)。
    onset = 该爪 action 首个长度≥MIN_SEG 的连续 <CLOSE_TH 段起点(回合内帧号);无则 None。
    返回的 act/st 按数据集全局行序排列,可直接切片当真值块用(★GT 以 parquet 为准,
    不走 feature_transform 往返——那条路出来的是 18 维映射口径,维号会错位)。"""
    files = sorted(glob.glob(f"{data_root}/data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{data_root}/data/chunk-*/file-*.parquet 一个都没有")
    df = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
         for f in files], ignore_index=True)
    act = np.stack(df["action"].apply(lambda a: np.asarray(a)).to_list())   # (N,20)
    st = np.stack(df["observation.state"].apply(lambda a: np.asarray(a)).to_list())
    eps = df["episode_index"].to_numpy()

    def first_close_onset(vals):
        below = vals < CLOSE_TH
        i, n = 0, len(vals)
        while i < n:
            if below[i]:
                j = i
                while j < n and below[j]:
                    j += 1
                if j - i >= MIN_SEG:
                    return i
                i = j
            else:
                i += 1
        return None

    out = {}
    for ep in np.unique(eps):
        idx = np.flatnonzero(eps == ep)
        a = act[idx]
        out[int(ep)] = (first_close_onset(a[:, GRIP_L]),
                        first_close_onset(a[:, GRIP_R]),
                        int(idx[0]), len(a))
    return out, act, st


def episode_bounds(dataset, ep: int) -> tuple[int, int]:
    """回合的全数据集行号区间(兼容 lerobot v2/v3 API,同 open_loop_eval)。"""
    if LEROBOT_API == "v2":
        return (int(dataset.episode_data_index["from"][ep]),
                int(dataset.episode_data_index["to"][ep]))
    return (int(dataset.meta.episodes[ep]["dataset_from_index"]),
            int(dataset.meta.episodes[ep]["dataset_to_index"]))


def main() -> None:
    args = parse_args()
    os.chdir(str(PROJECT_ROOT))   # robot_config/norm_stats 按项目根解析

    # 1) 策略服务:与 rollout 同一条推理路径,但 chunk_ret=True(一次拿整块 50 帧)
    print(f"[load] checkpoint : {args.checkpoint}")
    policy = LingbotVLAv2Server(
        args.checkpoint,
        robot_norm_path=args.norm_path,
        use_length=50, chunk_ret=True,
        use_bf16=True, use_fp32=False, use_compile=False,
    )
    policy.reset("h1")

    # 2) 闭爪锚点(parquet 真值)+ 全量真值动作
    onsets, act, _st = episode_grip_onsets(args.data)
    n_both = sum(1 for _, (ol, orr, _, _) in onsets.items()
                 if ol is not None and orr is not None)
    print(f"[锚点] {len(onsets)} 回合中 {n_both} 回合双爪都有闭合段")

    # 3) 数据集(带 action delta_timestamps → 样本自带真值 50 帧块,同 open_loop_eval)
    meta = LeRobotDatasetMetadata(Path(args.data).name, root=Path(args.data))
    action_features = policy.vla.feature_transform.org_features["actions"]
    delta = {af: [t / meta.fps for t in range(policy.config.chunk_size)]
             for af in action_features}
    dataset = LeRobotDataset(Path(args.data).name, root=Path(args.data),
                             delta_timestamps=delta)
    print(f"[load] 数据集 {len(dataset)} 帧 | fps={meta.fps} | chunk={policy.config.chunk_size}")

    # 4) 逐回合 × 双爪锚点 × 偏移,教师强制推理
    #    预测维数自适应:infer 返回 18 维映射口径(腰部被丢,爪在 16/17;
    #    rollout 里那条 18→20 防御分支实际每次都在走),20 维才是数据集口径
    grip_idx = None
    rows = []
    for ep in args.episodes:
        if ep not in onsets:
            print(f"[ep{ep}] 不存在,跳过")
            continue
        on_L, on_R, row0, n_frames = onsets[ep]
        start, end = episode_bounds(dataset, ep)
        if start != row0:
            raise RuntimeError(f"ep{ep} parquet 首行 {row0} ≠ 数据集首行 {start},行序对不上")
        policy.reset("h1")   # 清队列(chunk_ret=True 下每帧都真前向,保险起见仍重置)
        print(f"\n===== 回合 {ep}(帧数 {n_frames},锚点 L@{on_L} R@{on_R})=====")
        for side, onset, dim in (("L", on_L, GRIP_L), ("R", on_R, GRIP_R)):
            if onset is None:
                print(f"  [ep{ep}] {side} 爪无闭合段,跳过")
                continue
            for off in args.offsets:
                # 探测帧(回合内),尾部留出 50 帧块的空间
                f = int(np.clip(onset + off, 0, n_frames - policy.config.chunk_size - 1))
                traj, _ = prepare_eval_observation(policy, dataset[start + f])
                preds = policy.infer(traj)
                pred = np.concatenate(
                    [to_numpy(preds[af]) for af in action_features], axis=-1)  # (chunk, dim)
                if grip_idx is None:
                    grip_idx = ((GRIP_L, GRIP_R) if pred.shape[1] >= 20
                                else (GRIP_L - 2, GRIP_R - 2))
                    print(f"  [空间] 预测动作 {pred.shape[1]} 维 → 爪维号 {grip_idx}")
                # GT 直接切 parquet(20 维数据集口径,与预测同名侧对齐)
                gt = act[start + f: start + f + policy.config.chunk_size]
                row = dict(
                    ep=ep, side=side, anchor=onset, offset=off, frame=f,
                    gt_L=float(gt[:, GRIP_L].min()), gt_R=float(gt[:, GRIP_R].min()),
                    pred_L=float(pred[:, grip_idx[0]].min()),
                    pred_R=float(pred[:, grip_idx[1]].min()),
                )
                rows.append(row)
                tgt = row["pred_L"] if side == "L" else row["pred_R"]
                tag = "闭 ✓" if tgt < CLOSE_TH else "开 ✗"
                print(f"  [{side}锚 f{onset:4d}{off:+4d}] GTmin L={row['gt_L']:.2f} "
                      f"R={row['gt_R']:.2f} | 预测min L={row['pred_L']:.2f} "
                      f"R={row['pred_R']:.2f} → {side}爪:{tag}")

    if not rows:
        print("没有任何探测点,结束")
        return

    # 5) 汇总 + CSV
    with open(args.csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n===== 汇总:目标爪预测块内最小值(闭=<0.5)=====")
    for side, dim_name in (("L", "左爪"), ("R", "右爪")):
        print(f"-- {dim_name}锚点 --")
        for off in args.offsets:
            sel = [r for r in rows if r["side"] == side and r["offset"] == off]
            if not sel:
                continue
            tgt = np.array([r["pred_L"] if side == "L" else r["pred_R"] for r in sel])
            other = np.array([r["pred_R"] if side == "L" else r["pred_L"] for r in sel])
            print(f"  offset{off:+4d}: 目标爪 min 均值 {tgt.mean():.2f} "
                  f"(p10 {np.percentile(tgt, 10):.2f}) | 闭合率 {100 * (tgt < CLOSE_TH).mean():.0f}% "
                  f"| 另一侧爪 min 均值 {other.mean():.2f}(串行剧本检查)")
    print(f"\n[CSV] {args.csv}")
    print("[判读] anchor(offset 0)目标爪闭合率高 → 会闭=闭环分布漂移(结论A);"
          "一律≥0.85 → 未学会(结论B);只在正偏移闭 → 相位滞后")


if __name__ == "__main__":
    main()
