#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rollout_oracle_handoff.py —— 定界实验 C:抓取发生之后,模型会不会?
====================================================================
【背景】60k 判词:闭环 0% 缺口定位在"最后 5cm 对位"(映射✓渲染域✓发起✓
趋近✓,停短数厘米,方块从未进入指间构型)。在投钱修对位(租 A100 gbs4 /
v3 末段窗口)之前,必须先定界:**假设对位修好了、抓取真的发生了,后面的
"提起→搬运→放桶→松爪"模型会不会?**——若不会,修好对位也到不了成功,
A/B 的预期收益要重新估。

【做法】两种模式:
  1. handoff(主力)——重放交棒:
     闭环环境里开环重放演示动作到 F 帧(重放保真实测 0.001 rad,抓取
     物理真实发生,无任何焊接/瞬移作弊),然后 policy.reset 交棒,模型
     从演示中途接管,按与 Lingbot_H1_Rollout 完全相同的执行口径
     (默认 E 配置 gain1.2 + reinfer25)跑到上限或双块入桶。
       F = 指定臂的闭爪锚点 + offset:
         offset<0 → 轨迹内接管:模型能不能把最后几厘米走完并闭爪
         offset>0 → 抓后接管:方块真实在手里,模型会不会提/搬/放
  2. force_close(对照,C 的字面原设计):
     全程模型控制,指端距块 <8cm 时强制该侧爪闭合——看停位偏后的
     悬停状态下,闭爪能不能建立接触抓取(预期抓空,量化用)。

【判读】
  +15/+60:方块在手的接管。搬运/入桶完成 → 只欠对位,A/B 值得投;
           松爪掉块/不搬运 → 抓后段也没学会,修对位也到不了 100%
  -15/-60/-120:轨迹内接管。走完末段并闭爪 → 缺口=早期漂移累积(执行侧);
           仍停短 → 末段技能本身缺失(训练侧,支持加压/换配方)

【用法】(lingbotv2 环境,或直接跑——有解释器自纠正)
  python mjq_lingbotvla_v2/rollout_oracle_handoff.py                 # handoff R 全网格
  python mjq_lingbotvla_v2/rollout_oracle_handoff.py --arm L --offsets 15 60
  python mjq_lingbotvla_v2/rollout_oracle_handoff.py --mode force_close   # 对照组
====================================================================
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ============ 解释器自纠正保险(同训练/评估/rollout 启动器) ============
# ★必须放在 import torch/numpy 之前:base 环境没有这些包
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}", flush=True)
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# =============================================================

# 无头离屏渲染;必须在 import mujoco/cv2 之前(经 Lingbot_H1_Rollout 间接 import)
# --live 观看模式要开被动 viewer,不能用 egl(留默认 glx,同 rollout 观看模式)
if "--live" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 同目录探针模块
sys.path.insert(0, str(PROJECT_ROOT))                      # deploy.lingbotvla

# Lingbot_H1_Rollout import 时会解析默认参数,先把 argv 藏起来(同 grip_replay_probe)
_saved_argv = sys.argv
sys.argv = [_saved_argv[0]]
import Lingbot_H1_Rollout as R  # noqa: E402
sys.argv = _saved_argv
import mujoco  # noqa: E402
import cv2  # noqa: E402

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # noqa: E402
from grip_teacher_forced_probe import (  # noqa: E402
    CLOSE_TH, MIN_SEG, GRIP_L, GRIP_R, episode_grip_onsets,
)

DEFAULT_CKPT = PROJECT_ROOT / "output/h1_v4_full/checkpoints/global_step_60000/hf_ckpt"
DEFAULT_DATA = PROJECT_ROOT / "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"
DEFAULT_NORM = PROJECT_ROOT / "assets/norm_stats/h1_v4.json"

FORCE_CLOSE_VAL = 0.0          # 强制闭目标(demo 闭合即 0.0;ctrl 会被 clip 进 [0,1])
CUBE_OF_SIDE = {"L": "red_cube_left", "R": "red_cube_right"}
GRIP_DIM = {"L": GRIP_L, "R": GRIP_R}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="定界实验C:重放交棒/强制闭爪")
    p.add_argument("--mode", choices=["handoff", "force_close"], default="handoff")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--norm-path", type=str, default=str(DEFAULT_NORM))
    p.add_argument("--arm", choices=["R", "L"], default="R",
                   help="handoff 模式:以哪条臂的闭爪锚点为基准(R=先抓,L=后抓)")
    p.add_argument("--episodes", type=int, nargs="+",
                   default=[5, 55, 105, 155, 205, 255],
                   help="要重放的回合号(数据集 episode_index)")
    p.add_argument("--offsets", type=int, nargs="+", default=[-120, -60, -15, 15, 60],
                   help="handoff 模式:相对闭爪锚点的接管帧偏移")
    p.add_argument("--policy-steps", type=int, default=0,
                   help="交棒后模型最大帧数(0=按演示'锚点→松爪'时长自适应+150)")
    p.add_argument("--action-gain", type=float, default=1.2,
                   help="幅值增益(默认 1.2 = E 配置,与 60k 闭环评估一致)")
    p.add_argument("--reinfer-every", type=int, default=25,
                   help="重推间隔(默认 25 = E 配置)")
    p.add_argument("--lead-clip", type=float, default=0.0)
    p.add_argument("--force-close-dist", type=float, default=0.08,
                   help="force_close 模式:指端距块小于该值时强制闭爪(m)")
    p.add_argument("--settle", type=float, default=0.6)
    p.add_argument("--live", action="store_true",
                   help="观看模式:MuJoCo 主窗(头部视角)+腕部小窗;重放段快进,交棒后 30Hz 实时")
    p.add_argument("--max-steps", type=int, default=1800,
                   help="force_close 模式的每回合帧数上限(同闭环 rollout)")
    p.add_argument("--csv", type=str, default="",
                   help="逐帧日志 CSV(默认 /tmp/oracle_{mode}_{arm}.csv)")
    return p.parse_args()


def grasp_pt(model, data, side: str) -> np.ndarray:
    """双侧指环中点(与 rollout 的 ee 追踪同款口径)。side ∈ {'left','right'}。"""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                             f"{side}_omnipicker_hand_{k}_loop_Link")
           for k in ("narrow", "wide")]
    return np.mean([data.xpos[i] for i in ids], axis=0)


def first_reopen_onset(vals: np.ndarray, close_at: int) -> int | None:
    """闭爪锚点之后,首个长度≥MIN_SEG 的 ≥CLOSE_TH 段起点(demo 的松爪/放置时刻)。"""
    above = vals >= CLOSE_TH
    i = close_at
    n = len(vals)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= MIN_SEG:
                return i
            i = j
        else:
            i += 1
    return None


def snapshot(env: "R.H1RolloutEnv", model, data) -> dict:
    """当前时刻的诊断量:双侧指端-方块距离、方块位置、入桶标志。"""
    d_fing, cube_pos, in_bin = {}, {}, {}
    for side_key, cube in zip(("L", "R"), R.CUBE_BODY_NAMES):
        pt = grasp_pt(model, data, "left" if side_key == "L" else "right")
        d_fing[side_key] = float(np.linalg.norm(pt - env.cube_positions()[cube]))
    for cube in R.CUBE_BODY_NAMES:
        pos = env.cube_positions()[cube]
        cube_pos[cube] = pos
        in_bin[cube] = env.cube_in_bin(cube)
    return {"d_fing": d_fing, "cube": cube_pos, "in_bin": in_bin}


def frame_row(mode, arm, ep, offset, F, f, action_np, snap) -> list:
    cL, cR = snap["cube"]["red_cube_left"], snap["cube"]["red_cube_right"]
    return [
        mode, arm, ep, offset, F, f,
        float(action_np[GRIP_L]), float(action_np[GRIP_R]),
        snap["d_fing"]["L"], snap["d_fing"]["R"],
        cL[0], cL[1], cL[2], cR[0], cR[1], cR[2],
        int(snap["in_bin"]["red_cube_left"]), int(snap["in_bin"]["red_cube_right"]),
    ]


def replay_prefix(env, model, data, demo_act, F: int, viewer=None) -> None:
    """开环重放前 F 帧(与数采/rollout 同节奏;无 viewer 时无渲染全速)。"""
    frame_duration = 1.0 / R.FPS
    for f in range(F):
        env.apply_action(demo_act[f], env.home_ctrl[:2])
        target = data.time + frame_duration
        while data.time < target - 1e-9:
            mujoco.mj_step(model, data)
        if viewer is not None and viewer.is_running():
            if f % 30 == 0:
                print(f"    [重放快进] {f}/{F}")
            viewer.sync()


def policy_phase(env, policy, model, data, mode, arm, ep, offset, F,
                 steps, args, rows, viewer=None, live=False) -> dict:
    """交棒后的闭环段:与 Lingbot_H1_Rollout.run_episode 同一条执行路径
    (18→20 防御、NaN 保护、gain/lead-clip、reinfer 重置),外加逐帧日志。"""
    policy.reset(R.ROBO_NAME)
    last_action = env.home_ctrl.copy()
    frame_duration = 1.0 / R.FPS
    t0 = time.perf_counter()
    t_wall = time.monotonic()   # live 模式 30Hz 实时节奏的基准

    for step in range(steps):
        if args.reinfer_every < policy.use_length and step > 0 and step % args.reinfer_every == 0:
            policy.reset(R.ROBO_NAME)

        images = env.render_images()
        if live and R.show_wrist_windows(images):
            print(f"    [f{step}] 腕部窗口按 q,提前结束本段")
            break
        state = env.get_state()
        obs = {**images, "observation.state": state, "task": R.DEFAULT_TASK}

        preds = policy.infer(obs)
        action_np = np.asarray(preds["action"], dtype=np.float32).reshape(-1)
        if action_np.shape[0] == 18:      # 防御:18 维映射口径补回腰部(infer 实际每次都走这条)
            action_np = np.concatenate([env.home_ctrl[:2], action_np])
        if not np.all(np.isfinite(action_np)):
            action_np = last_action
        else:
            if args.action_gain != 1.0:
                st = np.asarray(state, dtype=np.float32).reshape(-1)
                action_np = st + args.action_gain * (action_np - st)
            if args.lead_clip > 0.0:
                st = np.asarray(state, dtype=np.float32).reshape(-1)
                action_np = st + np.clip(action_np - st, -args.lead_clip, args.lead_clip)
            if mode == "force_close":     # 对照:贴块强制闭(不闩锁,离开阈值即还给模型)
                snap_now = snapshot(env, model, data)
                for side_key in ("L", "R"):
                    if snap_now["d_fing"][side_key] < args.force_close_dist:
                        action_np[GRIP_DIM[side_key]] = FORCE_CLOSE_VAL
                action_np = np.asarray(action_np, dtype=np.float32)

        last_action = action_np
        env.apply_action(action_np, env.home_ctrl[:2])
        target = data.time + frame_duration
        while data.time < target - 1e-9:
            mujoco.mj_step(model, data)

        snap = snapshot(env, model, data)
        rows.append(frame_row(mode, arm, ep, offset, F, step, action_np, snap))
        if viewer is not None:
            if not viewer.is_running():
                print("    [view] 主窗已关,提前结束本段")
                break
            viewer.sync()
        if live:   # 30Hz 实时节奏(推理帧会略卡,同 rollout 观看模式)
            overrun = time.monotonic() - t_wall - (step + 1) * frame_duration
            if overrun < 0:
                time.sleep(-overrun)
        if all(snap["in_bin"].values()):
            break
        if (step + 1) % 150 == 0:
            print(f"    [f{step + 1}/{steps}] 爪cmd L{action_np[GRIP_L]:.2f} R{action_np[GRIP_R]:.2f}"
                  f" | 指距 L{snap['d_fing']['L']:.3f} R{snap['d_fing']['R']:.3f}"
                  f" | 入桶 {''.join('✓' if v else '✗' for v in snap['in_bin'].values())}")

    wall = time.perf_counter() - t0
    return {"wall_s": wall, "steps_run": len(rows)}


def main() -> None:
    args = parse_args()
    os.chdir(str(PROJECT_ROOT))

    csv_path = args.csv or f"/tmp/oracle_{args.mode}_{args.arm}.csv"
    print(f"[load] checkpoint : {args.checkpoint}")
    print(f"[load] 模式      : {args.mode} | 臂 {args.arm if args.mode == 'handoff' else '-'}"
          f" | gain {args.action_gain} | reinfer {args.reinfer_every}(E 配置)")
    policy = LingbotVLAv2Server(
        args.checkpoint,
        robot_norm_path=args.norm_path,
        use_length=50, chunk_ret=False,
        use_bf16=True, use_fp32=False, use_compile=False,
    )
    policy.reset(R.ROBO_NAME)

    onsets, act, _st = episode_grip_onsets(args.data)
    print(f"[锚点] {len(onsets)} 回合已读")

    model = mujoco.MjModel.from_xml_path(str(R.MJCF_PATH))
    model.vis.headlight.ambient = [0.4] * 3   # 头灯对齐数采(同 rollout main)
    model.vis.headlight.diffuse = [0.8] * 3
    model.vis.headlight.specular = [0.6] * 3
    data = mujoco.MjData(model)
    env = R.H1RolloutEnv(model, data, spawn_mode="home")
    print(f"[load] 场景: {R.MJCF_PATH.name} | task: {R.DEFAULT_TASK}")

    viewer_ctx = None
    if args.live:
        import mujoco.viewer as mujoco_viewer
        viewer_ctx = mujoco_viewer.launch_passive(model, data)
        viewer_ctx.__enter__()
        viewer_ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer_ctx.cam.fixedcamid = env.camera_ids["head_rgb"]
        env.ensure_renderers()   # viewer 起来之后再建离屏渲染器(GL 上下文顺序,同 rollout)
        for title in R.WRIST_VIEWER_TITLES.values():
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        print("[view] MuJoCo 主窗(头部视角)+ 左右腕小窗已启动;腕部小窗按 q 跳过本段,Ctrl+C 退出")

    rows: list[list] = []
    summary: list[dict] = []
    for ep in args.episodes:
        if ep not in onsets:
            print(f"[ep{ep}] 不存在,跳过")
            continue
        on_L, on_R, row0, n = onsets[ep]
        demo_act = act[row0: row0 + n]
        anchor = {"L": on_L, "R": on_R}[args.arm]
        if anchor is None:
            print(f"[ep{ep}] {args.arm} 爪无闭合段,跳过")
            continue
        release = first_reopen_onset(demo_act[:, GRIP_DIM[args.arm]], anchor)
        if release is None:
            release = n   # 没找到松爪段 → 按"打到回合尾"给预算

        grid = args.offsets if args.mode == "handoff" else [0]
        for offset in grid:
            F = int(np.clip(anchor + offset, 1, n - 2)) if args.mode == "handoff" else 0
            env.reset_episode(args.settle)
            spawn = snapshot(env, model, data)["cube"]      # 出生位(重放前)
            spawn_pos = {c: spawn[c].copy() for c in R.CUBE_BODY_NAMES}

            replay_prefix(env, model, data, demo_act, F, viewer=viewer_ctx)

            snap0 = snapshot(env, model, data)
            lift0 = {c: float(snap0["cube"][c][2] - spawn_pos[c][2]) for c in R.CUBE_BODY_NAMES}
            hand_off_in_hand = (lift0["red_cube_left" if args.arm == "L" else "red_cube_right"] > 0.03
                                and snap0["d_fing"][args.arm] < 0.12)
            if args.mode == "force_close":
                steps = args.max_steps
            elif args.policy_steps > 0:
                steps = args.policy_steps
            else:
                span = release - F if args.mode == "handoff" else n
                steps = int(np.clip(span + 150, 250, 750))
            act_at_handoff = demo_act[F - 1] if F > 0 else env.home_ctrl
            rows.append(frame_row(args.mode, args.arm, ep, offset, F, -1,
                                  act_at_handoff, snap0))    # f=-1:交棒时刻快照(最后一帧重放命令)
            print(f"\n===== ep{ep} {args.mode}{'@' + args.arm if args.mode == 'handoff' else ''}"
                  f" offset{offset:+d} → F={F}(锚{anchor},松爪@{release}) =====")
            print(f"  [交棒] {args.arm}指距 {snap0['d_fing'][args.arm]:.3f}m | "
                  f"块离台高度 L{lift0['red_cube_left']:+.3f} R{lift0['red_cube_right']:+.3f}"
                  f" | 在手判定 {'✓' if hand_off_in_hand else '✗'} | 模型预算 {steps} 帧")
            phase_rows_before = len(rows)
            policy_phase(env, policy, model, data, args.mode, args.arm, ep,
                         offset, F, steps, args, rows,
                         viewer=viewer_ctx, live=args.live)

            # ---- 本run汇总(从逐帧行里提) ----
            run_rows = rows[phase_rows_before:]
            arm_cube = CUBE_OF_SIDE[args.arm]
            col = {"red_cube_left": (10, 11, 12), "red_cube_right": (13, 14, 15)}[arm_cube]
            zs = np.array([r[col[2]] for r in run_rows])
            bin_xy = np.array([np.hypot(r[col[0]] - 0.7, r[col[1]]) for r in run_rows])
            grip = np.array([r[6] if args.arm == "L" else r[7] for r in run_rows])
            d_f = np.array([r[8] if args.arm == "L" else r[9] for r in run_rows])
            in_bin = max(r[16 if args.arm == "L" else 17] for r in run_rows) if run_rows else 0
            z_drop = float(zs.max() - zs[-1]) if len(zs) else 0.0
            summary.append(dict(
                ep=ep, offset=offset, F=F, in_hand0=hand_off_in_hand,
                lift0=lift0[arm_cube], steps=len(run_rows),
                grip_min=float(grip.min()) if len(grip) else float("nan"),
                grip_mean=float(grip.mean()) if len(grip) else float("nan"),
                d_fing_min=float(d_f.min()) if len(d_f) else float("nan"),
                cube_zmax=float(zs.max()) if len(zs) else float("nan"),
                z_drop=z_drop, bin_xy_min=float(bin_xy.min()) if len(bin_xy) else float("nan"),
                ever_in_bin=bool(in_bin),
            ))
            s = summary[-1]
            print(f"  [结果] {len(run_rows)} 帧"
                  f" | 爪cmd min {s['grip_min']:.2f} | 指距min {s['d_fing_min']:.3f}"
                  f" | 块z max {s['cube_zmax']:+.3f} 掉落 {s['z_drop']:+.3f}"
                  f" | 块-桶xy min {s['bin_xy_min']:.3f} | 曾入桶 {'✓' if s['ever_in_bin'] else '✗'}")

    if rows:
        with open(csv_path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["mode", "arm", "ep", "offset", "F", "f",
                        "grip_cmd_L", "grip_cmd_R", "d_fing_L", "d_fing_R",
                        "cL_x", "cL_y", "cL_z", "cR_x", "cR_y", "cR_z", "in_L", "in_R"])
            w.writerows(rows)
        print(f"\n[CSV] {csv_path}({len(rows)} 行)")

    print("\n===== 汇总(每 run 一行)=====")
    print(f"{'ep':>4} {'off':>5} {'F':>5} {'在手':>4} {'爪min':>6} {'指距min':>8} "
          f"{'zmax':>7} {'掉落':>7} {'桶xy':>7} {'入桶':>4}")
    for s in summary:
        print(f"{s['ep']:>4} {s['offset']:>+5} {s['F']:>5} {'✓' if s['in_hand0'] else '✗':>4} "
              f"{s['grip_min']:>6.2f} {s['d_fing_min']:>8.3f} {s['cube_zmax']:>+7.3f} "
              f"{s['z_drop']:>+7.3f} {s['bin_xy_min']:>7.3f} {'✓' if s['ever_in_bin'] else '✗':>4}")
    print("[判读] +offset:在手接管后搬运/入桶→只欠对位;松爪掉块/不搬→抓后段也没学会\n"
          "       -offset:轨迹内接管仍停短→末段技能缺失(训练侧);能走完→缺口=早期漂移(执行侧)")

    if viewer_ctx is not None:
        viewer_ctx.__exit__(None, None, None)
        R.close_wrist_windows()


if __name__ == "__main__":
    main()
