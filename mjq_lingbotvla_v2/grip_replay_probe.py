#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grip_replay_probe.py —— 演示重放 + 中途推理(闭环渲染域 vs 状态漂移判决)
====================================================================
【背景】教师强制探针(grip_teacher_forced_probe.py)已证实:喂演示观测,
模型在闭爪时刻 100% 正确闭爪且分侧 → 抓取映射已学会。但那用的是
**数据集里存的数采渲染图像**。闭环 rollout 用的是**我们自己的渲染**
(头灯已对齐,但同 pose 下像素级仍可能有差)。剩下最后一个混淆:
  A. 纯状态漂移:闭环臂停 6-10cm,"贴块"视觉条件从未出现(重放到位就会闭)
  B. 渲染域差:我们的渲染本身让模型认不出"贴块"(重放到位也不闭)

【做法】闭环环境里**开环重放演示动作**(不走模型控制,从同一 home/出生点起,
逐帧写演示 ctrl),臂因此走到演示位姿(~0-2cm 级,vs 闭环失败的 6-10cm);
在闭爪锚点前后若干帧,用**我们 rollout 的渲染**喂模型,看爪闭不闭。
同时逐探针帧对账:重放 state vs 演示 state(保真度)、指端-方块距离。

【判读】anchor 处目标爪闭合率:
  与教师强制相当(≈100%) → 渲染域没问题 → 纯状态漂移,续训方向明确
  显著掉(≤50%)        → 渲染域有差 → 先修域再训练

【用法】python mjq_lingbotvla_v2/grip_replay_probe.py
  # --episodes 5 55 105 ... | --offsets -45 -30 -15 0 15 | --csv /tmp/xxx.csv
====================================================================
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# ============ 解释器自纠正保险(同训练/评估/rollout 启动器) ============
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}", flush=True)
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# =============================================================

# 无头离屏渲染;必须在 import mujoco/cv2 之前(经 Lingbot_H1_Rollout 间接 import)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 同目录探针模块
sys.path.insert(0, str(PROJECT_ROOT))                      # deploy.lingbotvla

# ---- 复用 Lingbot_H1_Rollout 的环境实现(该模块 import 时会解析默认参数,
#      先把 argv 藏起来,拿默认值;它只定义类/常量,不跑 main)----
_saved_argv = sys.argv
sys.argv = [_saved_argv[0]]
import Lingbot_H1_Rollout as R  # noqa: E402
sys.argv = _saved_argv
import mujoco  # noqa: E402  (上面已随 R 导入,这里拿名字直用)

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # noqa: E402
from grip_teacher_forced_probe import (  # noqa: E402
    CLOSE_TH, GRIP_L, GRIP_R, episode_grip_onsets,
)

DEFAULT_CKPT = PROJECT_ROOT / "output/h1_v4_full/checkpoints/global_step_40000/hf_ckpt"
DEFAULT_DATA = PROJECT_ROOT / "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"
DEFAULT_NORM = PROJECT_ROOT / "assets/norm_stats/h1_v4.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="演示重放+中途推理(渲染域 vs 状态漂移)")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--norm-path", type=str, default=str(DEFAULT_NORM))
    p.add_argument("--episodes", type=int, nargs="+",
                   default=[5, 55, 105, 155, 205, 255],
                   help="要重放的回合号(默认 6 回合铺满 0-301)")
    p.add_argument("--offsets", type=int, nargs="+", default=[-45, -30, -15, 0, 15],
                   help="相对闭爪锚点的探测帧偏移")
    p.add_argument("--settle", type=float, default=0.6,
                   help="回合开始前静置秒数(与 rollout/数采同款)")
    p.add_argument("--csv", type=str, default="/tmp/grip_replay_probe.csv")
    return p.parse_args()


def grasp_pt(model, data, side: str) -> np.ndarray:
    """双侧指环中点(与 rollout 的 ee 追踪同款口径)。"""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             f"{side}_omnipicker_hand_{k}_loop_Link")
           for k in ("narrow", "wide")]
    return np.mean([data.xpos[i] for i in ids], axis=0)


def main() -> None:
    args = parse_args()
    os.chdir(str(PROJECT_ROOT))

    # 1) 策略(与 rollout 同参数,但 chunk_ret=True:每个探测点独立整块推理)
    print(f"[load] checkpoint : {args.checkpoint}")
    policy = LingbotVLAv2Server(
        args.checkpoint,
        robot_norm_path=args.norm_path,
        use_length=50, chunk_ret=True,
        use_bf16=True, use_fp32=False, use_compile=False,
    )
    policy.reset("h1")

    # 2) 演示真值(锚点 + action/state 全量,parquet 口径)
    onsets, act, st = episode_grip_onsets(args.data)
    print(f"[锚点] {len(onsets)} 回合")

    # 3) 闭环环境:与 rollout main() 同款场景 + 头灯对齐(数采 teleop 调亮过,
    #    xml 默认值会造成训练/推理图像域差距——见 memory: headlight-domain-gap)
    model = mujoco.MjModel.from_xml_path(str(R.MJCF_PATH))
    model.vis.headlight.ambient = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
    model.vis.headlight.specular = [0.6, 0.6, 0.6]
    data = mujoco.MjData(model)
    env = R.H1RolloutEnv(model, data, spawn_mode="home")
    print(f"[load] 场景: {R.MJCF_PATH.name} | task: {R.DEFAULT_TASK}")

    frame_duration = 1.0 / R.FPS
    rows = []
    for ep in args.episodes:
        if ep not in onsets:
            print(f"[ep{ep}] 不存在,跳过")
            continue
        on_L, on_R, row0, n = onsets[ep]
        demo_act = act[row0: row0 + n]        # (n,20) 演示 ctrl
        demo_st = st[row0: row0 + n]
        probes = sorted({int(np.clip(o + off, 0, n - 51))
                         for o, off in ((on_L, x) for x in args.offsets)
                         if o is not None}
                        | {int(np.clip(o + off, 0, n - 51))
                           for o, off in ((on_R, x) for x in args.offsets)
                           if o is not None})
        # 帧号 → (侧, 锚点, 偏移) 标签(两锚点探测帧几乎不会撞车)
        tag_of = {}
        for side_name, o in (("L", on_L), ("R", on_R)):
            if o is None:
                continue
            for off in args.offsets:
                f = int(np.clip(o + off, 0, n - 51))
                tag_of.setdefault(f, (side_name, o, off))

        env.reset_episode(args.settle)
        policy.reset("h1")
        print(f"\n===== 回合 {ep}(重放 {n} 帧,锚点 L@{on_L} R@{on_R},探测 {len(probes)} 点)=====")
        for f in range(n):
            if f in tag_of:
                side, anchor, off = tag_of[f]
                images = env.render_images()
                obs_state = env.get_state()
                preds = policy.infer({**images, "observation.state": obs_state,
                                      "task": R.DEFAULT_TASK})
                pred = np.asarray(preds["action"])          # (chunk, 18 或 20) 映射口径
                g_l, g_r = ((GRIP_L, GRIP_R) if pred.shape[1] >= 20
                            else (GRIP_L - 2, GRIP_R - 2))
                gt = demo_act[f: f + 50]                    # parquet 真值块(20 维)
                d_pt = {c: float(np.linalg.norm(
                    grasp_pt(model, data, s) - env.cube_positions()[c]))
                    for c, s in zip(R.CUBE_BODY_NAMES, ("left", "right"))}
                row = dict(
                    ep=ep, side=side, anchor=anchor, offset=off, frame=f,
                    gt_L=float(gt[:, GRIP_L].min()), gt_R=float(gt[:, GRIP_R].min()),
                    pred_L=float(pred[:, g_l].min()), pred_R=float(pred[:, g_r].min()),
                    dev_joint=float(np.abs(obs_state[2:20] - demo_st[f, 2:20]).mean()),
                    dist_L=d_pt["red_cube_left"], dist_R=d_pt["red_cube_right"],
                )
                rows.append(row)
                tgt = row["pred_L"] if side == "L" else row["pred_R"]
                tag = "闭 ✓" if tgt < CLOSE_TH else "开 ✗"
                print(f"  [{side}锚 f{anchor:4d}{off:+4d}] GTmin L={row['gt_L']:.2f} "
                      f"R={row['gt_R']:.2f} | 预测min L={row['pred_L']:.2f} "
                      f"R={row['pred_R']:.2f} → {side}爪:{tag} | 保真 "
                      f"Δjoint={row['dev_joint']:.3f} 指端距块 L{row['dist_L']:.2f}/R{row['dist_R']:.2f}")
            # 重放:写演示 ctrl,步进 1/30s(与数采/rollout 同节奏)
            env.apply_action(demo_act[f], env.home_ctrl[:2])
            target = data.time + frame_duration
            while data.time < target - 1e-9:
                mujoco.mj_step(model, data)

    if not rows:
        print("没有任何探测点,结束")
        return

    with open(args.csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    devs = np.array([r["dev_joint"] for r in rows])
    print(f"\n===== 汇总(重放保真:Δ关节 均值 {devs.mean():.3f} / p90 {np.percentile(devs, 90):.3f} rad)=====")
    for side, dim_name in (("L", "左爪"), ("R", "右爪")):
        print(f"-- {dim_name}锚点 --")
        for off in args.offsets:
            sel = [r for r in rows if r["side"] == side and r["offset"] == off]
            if not sel:
                continue
            tgt = np.array([r["pred_L"] if side == "L" else r["pred_R"] for r in sel])
            d_near = np.array([r["dist_L"] if side == "L" else r["dist_R"] for r in sel])
            print(f"  offset{off:+4d}: 目标爪 min 均值 {tgt.mean():.2f} "
                  f"| 闭合率 {100 * (tgt < CLOSE_TH).mean():.0f}% "
                  f"| 重放指端距目标块 {d_near.mean():.3f}m")
    print(f"\n[CSV] {args.csv}")
    print("[判读] anchor 闭合率≈教师强制(100%)→纯状态漂移,续训方向明确;"
          "显著掉(≤50%)→渲染域有差,先修域再训练")


if __name__ == "__main__":
    main()
