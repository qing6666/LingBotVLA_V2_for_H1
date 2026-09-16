#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_collected_data.py —— 数采数据离线自检(采集中不用开,采完批次后跑)
====================================================================
【什么时候跑】(不介入采集过程,只读已写好的 parquet)
  ① 采完第 1 条:验证夹爪斜坡真的生效(决定这批数据能不能用)
  ② 每采 20~30 条:抽最新几条复查
  ③ 收工:全量过一遍(--episodes 0 = 全部)
  ★ 跑之前先关掉遥操作程序,等后台编码线程把 parquet/视频写完再跑。

【检查项】(每项 ✓/✗,任何 ✗ 退出码=1,方便脚本化)
  1. 夹爪过渡:开→闭的过渡帧数(斜坡改后应 ~15 帧;旧阶跃数据=2 帧) +
     单帧最大跳变 ≤0.15(斜坡 2.0 行程/秒 @30fps → 每帧 ≈0.067)
  2. 闭合完整度:每次闭合都要扣到底(<0.15),放开也要放到底(>0.85)
     —— 抓"半闭合拖拽"的操作坏习惯
  3. 双臂全冻结占比:两边 14 个关节指令全不动的帧占比 ≤20%
     (旧数据 13%,超过说明操作停顿/离合挂机太多)
  4. 左右臂指令 jerk 相当:mean|二阶差分| 比值 <2
     (一边明显更抖 = 单侧操作习惯差,那一侧的偏差会进模型)
  5. 信息项(不判 ✓/✗):action-state 平均滞后、回合长度、汇总表

【怎么跑】(lingbotv2 环境)
  python mjq_lingbotvla_v2/check_collected_data.py                       # 默认查旧裁剪数据最新 3 条
  python mjq_lingbotvla_v2/check_collected_data.py --data <新数据目录> --episodes 3
  python mjq_lingbotvla_v2/check_collected_data.py --data <新数据目录> --episodes 0   # 全量
  加 --plot 额外存夹爪/右肩曲线图到 /tmp(目测 action 与 state 贴合度)
====================================================================
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# ============ 解释器自纠正保险(必须在 import numpy/pandas 之前) ============
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}", flush=True)
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# =========================================================================

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "H1_simulation_model" / "H1_build" / "data" / "h1_build_pick_v2_trimmed"

# 20 维动作布局(数采记录器口径):[0:2]腰 [2:9]左臂 [9:16]右臂 [16:18]头 [18:20]夹爪
L_ARM, R_ARM, GRIPS = list(range(2, 9)), list(range(9, 16)), (18, 19)
ARM_DIMS = L_ARM + R_ARM

# ---- 阈值区(改动要有依据)----
# 过渡检测的是 >0.7→<0.3 区段(0.4 行程):斜坡 2.0 行程/s → 0.4/2.0*30fps ≈ 6 帧,
# 留余量取 (4,20);旧阶跃数据 = 1~2 帧,与斜坡数据完全可分。
GRIP_TRANSITION_OK = (4, 20)     # 开→闭过渡帧数合格区间(0.7→0.3 区段口径)
GRIP_MAX_JUMP = 0.15             # 单帧最大跳变(斜坡≈0.067,阶跃≈0.9)
GRIP_CLOSE_FLOOR = 0.15          # 闭合要扣到的深度
GRIP_OPEN_CEIL = 0.85            # 放开要放到的开度
FROZEN_MAX_FRAC = 0.20           # 双臂全冻结占比上限(旧数据典型 8~15%)
STUTTER_MAX_RUN = 120            # 单段连续全冻结帧数上限(4s;超了=流卡死,非正常停顿)
JERK_RATIO_MAX = 2.0             # 左右 jerk 之比上限


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数采数据离线自检")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                        help=f"数据集目录(默认={DEFAULT_DATA.name};新批次传新目录)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="抽最新几条检查(0=全量,收工时用)")
    parser.add_argument("--plot", action="store_true", help="存夹爪/右肩曲线图到 /tmp 供目测")
    return parser.parse_args()


def load_actions(data_dir: str) -> pd.DataFrame:
    pqs = sorted(glob.glob(os.path.join(data_dir, "data", "**", "*.parquet"), recursive=True))
    if not pqs:
        raise FileNotFoundError(f"{data_dir}/data 下没找到 parquet(确认目录对、程序已写完盘)")
    # 数据可能还在被后台编码线程写:10 秒内有改动就提醒
    newest = max(os.path.getmtime(p) for p in pqs)
    if time.time() - newest < 10:
        print("⚠ 最新 parquet 10 秒内还在变化 —— 先关掉遥操作程序等写盘结束再跑!\n")
    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    for col in ("action", "episode_index"):
        if col not in df.columns:
            raise KeyError(f"parquet 缺列 {col}(列: {list(df.columns)[:8]} ...)")
    if "observation.state" not in df.columns:
        print("⚠ parquet 没有 observation.state 列,滞后项将跳过")
    return df


def grip_close_events(g: np.ndarray) -> list:
    """返回 (闭合起始i, 过渡帧数, 闭合最深值) 列表。闭合=从>0.7 落到<0.3。"""
    events = []
    i = 0
    while i < len(g) - 1:
        if g[i] > 0.7 and g[i + 1] < 0.3:          # 旧阶跃数据:1 帧瞬跳
            j = i + 1
            events.append((i, 1, float(g[i + 1:].min() if len(g) > i + 1 else g[-1])))
            i = j
            continue
        if g[i] > 0.7 and g[i + 1] <= 0.7:          # 斜坡数据:渐变下降
            j = i + 1
            while j < len(g) and g[j] >= 0.3:
                j += 1
            if j < len(g):                          # 真正闭到 <0.3 才算一次闭合
                depth = float(g[i:j + 5].min()) if j + 5 <= len(g) else float(g[i:].min())
                events.append((i, j - i, depth))
                i = j
                continue
        i += 1
    return events


def check_episode(ep_id: int, a: np.ndarray, s: np.ndarray | None,
                  do_plot: bool) -> tuple[bool, list[str]]:
    """一条回合的 5 项检查。返回 (是否全过, 明细行)。"""
    ok, lines = True, []
    lines.append(f"━━ 回合 {ep_id}({len(a)} 帧)━━")

    # 1) 夹爪过渡 + 单帧跳变
    for dim, name in zip(GRIPS, ("左爪", "右爪")):
        g = a[:, dim]
        jump = float(np.abs(np.diff(g)).max()) if len(g) > 1 else 0.0
        events = grip_close_events(g)
        trans = [t for _, t, _ in events]
        verdict, detail = "✓", ""
        if not events:
            verdict, detail = "⚠", "本回合没有闭合事件(没抓东西?)"
        else:
            med = float(np.median(trans))
            if not (GRIP_TRANSITION_OK[0] <= med <= GRIP_TRANSITION_OK[1]):
                verdict = "✗"
            if jump > GRIP_MAX_JUMP:
                verdict = "✗"
                detail += f" 单帧跳变{jump:.2f}> {GRIP_MAX_JUMP}(阶跃特征!)"
            ok &= (verdict != "✗")
            detail = f" 过渡中位 {med:.0f} 帧(合格区间 {GRIP_TRANSITION_OK}){detail}"
        lines.append(f"  1.{name}  {verdict}{detail}")

        # 2) 闭合完整度
        if events:
            bad = [(i, t, d) for i, t, d in events if d > GRIP_CLOSE_FLOOR]
            if bad:
                lines.append(f"  2.{name}  ✗ 有 {len(bad)} 次闭合没扣到底(最深只到 "
                             f"{max(d for _, _, d in bad):.2f} > {GRIP_CLOSE_FLOOR})——半闭合拖拽会教坏模型")
                ok = False
            else:
                lines.append(f"  2.{name}  ✓ 每次闭合都扣到底(≤{GRIP_CLOSE_FLOOR})")
        # 放开完整度(信息项)
        opens = np.where((np.diff(g) > 0) & (g[1:] > 0.3) & (g[:-1] <= 0.3))[0]
        if len(opens) and g.max() < GRIP_OPEN_CEIL:
            lines.append(f"        ⚠ 放开最大只到 {g.max():.2f}(<{GRIP_OPEN_CEIL}),没放到底")

    # 3) 冻结/卡顿(只看中途):尾部"做完站定、再按 X 保存"的停留是正常操作节奏
    #    (v3 实测 42/154 条带 50~205 帧尾停),计入冻结会普遍误报;
    #    真卡顿特征 = 中途连续全僵 ≥120 帧 + 恢复跳变远超该回合基线(见 v3 ep33)。
    da = np.abs(np.diff(a[:, ARM_DIMS], axis=0))
    frozen_mask = da.max(axis=1) < 1e-3 if len(da) else np.array([])
    tail = 0                                    # 结尾连续全僵的长度(保存前停留)
    if len(frozen_mask) and frozen_mask[-1]:
        while tail < len(frozen_mask) and frozen_mask[-1 - tail]:
            tail += 1
    mid_mask = frozen_mask[: len(frozen_mask) - tail] if len(frozen_mask) else frozen_mask
    frozen = float(mid_mask.mean()) if len(mid_mask) else 0.0
    max_run, run = 0, 0
    for f in mid_mask:
        run = run + 1 if f else 0
        max_run = max(max_run, run)
    verdict = "✓" if frozen <= FROZEN_MAX_FRAC and max_run <= STUTTER_MAX_RUN else "✗"
    ok &= (verdict == "✓")
    note = "  ★卡顿嫌疑(整条丢弃)!" if max_run > STUTTER_MAX_RUN or frozen > FROZEN_MAX_FRAC else ""
    lines.append(f"  3.冻结/卡顿    {verdict} 中途冻结 {frozen:.1%}(上限 {FROZEN_MAX_FRAC:.0%})"
                 f" | 最长中途全僵 {max_run} 帧(上限 {STUTTER_MAX_RUN})"
                 f" | 尾部停留 {tail} 帧(保存节奏,不计){note}")

    # 4) 左右臂 jerk
    jl = float(np.abs(np.diff(a[:, L_ARM], axis=0, n=2)).mean())
    jr = float(np.abs(np.diff(a[:, R_ARM], axis=0, n=2)).mean())
    ratio = max(jl, jr) / max(min(jl, jr), 1e-9)
    verdict = "✓" if ratio < JERK_RATIO_MAX else "✗"
    ok &= (verdict == "✓")
    lines.append(f"  4.左右jerk     {verdict} 左 {jl:.5f} vs 右 {jr:.5f}(比值 {ratio:.2f} < {JERK_RATIO_MAX})")

    # 5) action-state 滞后(信息项)
    if s is not None and s.shape == a.shape:
        lag_arm = float(np.abs(a[:, 2:18] - s[:, 2:18]).mean())
        lines.append(f"  5.指令-状态滞后(信息项) 上身平均 |action-state| = {lag_arm:.4f} rad(平贴合;突然变大=那几段指令丢/卡)")

    if do_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, dim, name in zip(axes, (18, 19, 9), ("Grip.L cmd vs state", "Grip.R cmd vs state", "R_J1 cmd vs state")):
            ax.plot(a[:, dim], label="action", lw=1)
            if s is not None and s.shape == a.shape:
                ax.plot(s[:, dim], label="state", lw=1, alpha=0.7)
            ax.set_title(f"ep{ep_id} {name}")
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out = f"/tmp/check_collected_ep{ep_id}.png"
        fig.savefig(out, dpi=110)
        lines.append(f"  [plot] 曲线图已存 {out}")
    return ok, lines


def main() -> None:
    args = parse_args()
    print(f"[check] 数据目录: {args.data}")
    df = load_actions(args.data)
    acts = np.stack(df["action"].tolist())
    eps = df["episode_index"].to_numpy()
    states = None
    if "observation.state" in df.columns:
        states = np.stack(df["observation.state"].tolist())

    uniq = np.unique(eps)
    picked = uniq if args.episodes <= 0 else uniq[-args.episodes:]
    print(f"[check] 共 {len(uniq)} 条回合,检查其中 {len(picked)} 条:{list(picked)}\n")

    n_pass = 0
    for ep_id in picked:
        m = eps == ep_id
        ok, lines = check_episode(int(ep_id), acts[m], states[m] if states is not None else None, args.plot)
        n_pass += ok
        print("\n".join(lines))
        if not ok:
            print("  ↑ 这条有问题:整条丢弃(A 键复位重来),别让坏数据进训练集")
        print()

    n = len(picked)
    print("===== 自检汇总 =====")
    print(f"{n_pass}/{n} 条通过")
    if n_pass < n:
        print("结论:存在不合格回合 → 丢弃或整批判定不可用(若是斜坡没生效这类系统性问题)")
        sys.exit(1)
    print("结论:抽检全部通过,可以继续采/送训练")


if __name__ == "__main__":
    main()
