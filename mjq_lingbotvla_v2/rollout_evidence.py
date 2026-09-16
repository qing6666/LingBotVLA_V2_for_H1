#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理取证回合(按《排查方案》§五/§六/§七/§八/§十三/§十五)+ CSV 落盘
================================================================
复用 Lingbot_H1_Rollout 模块的 env/常量,无头跑 1 回合(默认策略,无 lead-clip),
逐帧记录 state/action/推理耗时,每次前向记录完整 50 步块,落盘 debug_actions/。
产出:
  ① 块内 |Δaction| 分布(臂维)
  ② 边界 |Δaction| 分布(前向帧 = 新块第0步 vs 上一执行动作)
  ③ 每块 chunk[0] 超前 state(方案§八)
  ④ 夹爪命令/状态时序(闭爪时机)
  ⑤ 推理耗时分布 + 冻结占空比(方案§十五)
  ⑥ action vs state 曲线数据(action_state.csv)+ 每块 chunk_XXX.csv
"""
import sys, os, importlib.util, time, json
from pathlib import Path

_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _PY and Path(_PY).exists():
    os.execv(_PY, [_PY] + sys.argv)

import numpy as np
import mujoco

PROJECT_ROOT = Path("/home/mjq/robot_item/lingbot-vla-v2-main")
ROLLOUT = PROJECT_ROOT / "mjq_lingbotvla_v2" / "Lingbot_H1_Rollout.py"
MAX_STEPS = 1200
OUT_DIR = Path("/tmp/debug_actions"); OUT_DIR.mkdir(exist_ok=True)

# ---- 加载 rollout 模块(ARGV 清空防 argparse 误解析) ----
sys.argv = [sys.argv[0]]
os.chdir(str(PROJECT_ROOT))
spec = importlib.util.spec_from_file_location("rollout_mod", ROLLOUT)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
print(f"[0] 模块加载 ✓ MJCF={R.MJCF_PATH.name} FPS={R.FPS} task={R.DEFAULT_TASK!r}")

# ---- 策略(与 main() 完全同参) ----
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
CKPT = PROJECT_ROOT / "output/h1_v4_w2/checkpoints/global_step_120000/hf_ckpt"
policy = LingbotVLAv2Server(str(CKPT), robot_norm_path=str(PROJECT_ROOT / "assets/norm_stats/h1_v4.json"),
                            use_length=50, chunk_ret=False, use_bf16=True, use_fp32=False, use_compile=False)
policy.reset(R.ROBO_NAME)

# ---- 场景(复刻 main:头灯对齐数采) ----
model = mujoco.MjModel.from_xml_path(str(R.MJCF_PATH))
model.vis.headlight.ambient = [0.4, 0.4, 0.4]
model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
model.vis.headlight.specular = [0.6, 0.6, 0.6]
data = mujoco.MjData(model)
env = R.H1RolloutEnv(model, data, spawn_mode="home")
env.reset_episode(1.0)
policy.reset(R.ROBO_NAME)
print("[0] 场景/策略就绪,开始取证回合(无 lead-clip,原始行为)")

ARM = list(range(2, 16))          # 数据集口径臂维
states, actions, infer_ms_all, fwd_frames = [], [], [], []
chunks = []                        # (frame, state20@前向, chunk50x18)
frame_duration = 1.0 / R.FPS
last_action = env.home_ctrl.copy()
t_start = time.perf_counter()

for step in range(MAX_STEPS):
    images = env.render_images()
    st = env.get_state()
    obs = {**images, "observation.state": st, "task": R.DEFAULT_TASK}
    t0 = time.perf_counter()
    preds = policy.infer(obs)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    action_np = np.asarray(preds["action"], dtype=np.float32).reshape(-1)
    if action_np.shape[0] == 18:
        action_np = np.concatenate([env.home_ctrl[:2], action_np])
    if not np.all(np.isfinite(action_np)):
        action_np = last_action
    did_forward = policy.global_step % policy.use_length == 1
    if did_forward:
        fwd_frames.append(step)
        infer_ms_all.append(infer_ms)
        chunk = policy.last_action_chunk["action"][0]     # (50,18)
        chunks.append((step, st.copy(), np.asarray(chunk, dtype=np.float32)))
    states.append(st.copy()); actions.append(action_np.copy())
    last_action = action_np
    env.apply_action(action_np, env.home_ctrl[:2])
    target = env.data.time + frame_duration
    while env.data.time < target - 1e-9:
        mujoco.mj_step(env.model, env.data)
    if (step + 1) % 300 == 0:
        print(f"  f{step+1}/{MAX_STEPS} | 前向 {len(fwd_frames)} | 累计 {(time.perf_counter()-t_start)/60:.1f} min")

wall_min = (time.perf_counter() - t_start) / 60
S = np.stack(states); A = np.stack(actions)
print(f"[1] 回合完成 {MAX_STEPS} 帧 | 墙钟 {wall_min:.1f} min | 前向 {len(fwd_frames)} 次")

# ---- 落盘(方案§五) ----
np.savetxt(OUT_DIR / "action_state.csv",
           np.concatenate([np.arange(MAX_STEPS)[:, None], S, A], axis=1),
           delimiter=",", header="frame," + ",".join([f"s{i}" for i in range(20)] + [f"a{i}" for i in range(20)]))
for i, (f0, st0, ch) in enumerate(chunks):
    np.savetxt(OUT_DIR / f"chunk_{i:03d}.csv", ch, delimiter=",",
               header=",".join(f"d{j}" for j in range(ch.shape[1])), comments="")
print(f"[2] CSV 落盘 → {OUT_DIR} (action_state.csv + {len(chunks)} 个 chunk)")

# ---- ① 块内 vs ② 边界 跳变(执行动作口径,臂维) ----
dA = np.abs(np.diff(A[:, ARM], axis=0)).max(axis=1)       # 帧 k→k+1 最大臂维跳变
is_bnd = np.zeros(len(dA), bool)
for f in fwd_frames:
    if 0 < f <= len(dA):
        is_bnd[f - 1] = True                              # f-1→f 即旧块末帧→新块第0步
within, boundary = dA[~is_bnd], dA[is_bnd]
pct = lambda v, q: float(np.percentile(v, q))
print(f"\n[①②] 臂维逐帧最大跳变(rad):")
print(f"    块内  n={len(within):4d}  中位 {pct(within,50):.3f}  p90 {pct(within,90):.3f}  max {within.max():.3f}")
print(f"    边界  n={len(boundary):3d}  中位 {pct(boundary,50):.3f}  p90 {pct(boundary,90):.3f}  max {boundary.max():.3f}"
      f"  → 中位倍数 {pct(boundary,50)/max(pct(within,50),1e-9):.1f}x")

# ---- ③ chunk[0] 超前 state(方案§八;数据集口径臂+爪) ----
leads, lead_frames = [], []
for f0, st0, ch in chunks:
    a0 = ch[0]                                            # 18 维:L7 R7 head2 grip2
    lead_arm = np.abs(a0[0:14] - st0[2:16]).max()
    leads.append(lead_arm); lead_frames.append(f0)
leads = np.array(leads); lead_frames = np.array(lead_frames)
for lab, mask in [("前段(f<300)", lead_frames < 300),
                  ("中段(300-700)", (lead_frames >= 300) & (lead_frames < 700)),
                  ("后段(f>=700)", lead_frames >= 700)]:
    if mask.sum():
        print(f"[③] chunk[0]超前state 臂维max: {lab} n={mask.sum():2d} 中位 {pct(leads[mask],50):.3f} max {leads[mask].max():.3f}")

# ---- ④ 夹爪时序 ----
gl_cmd, gl_st = A[:, 18], S[:, 18]
gr_cmd, gr_st = A[:, 19], S[:, 19]
first_close = lambda cmd: int(np.argmax(cmd < 0.5)) if (cmd < 0.5).any() else -1
print(f"\n[④] 夹爪(0.5 阈值): L命令首闭 f{first_close(gl_cmd)} L状态首闭 f{first_close(gl_st)}"
      f" | R命令首闭 f{first_close(gr_cmd)} R状态首闭 f{first_close(gr_st)}")
print(f"    末帧: L cmd {gl_cmd[-1]:.2f}/st {gl_st[-1]:.2f} | R cmd {gr_cmd[-1]:.2f}/st {gr_st[-1]:.2f}")

# ---- ⑤ 推理耗时 + 冻结占空比 ----
im = np.array(infer_ms_all)
per_cycle = 50 * 1000.0 / R.FPS
print(f"\n[⑤] 前向耗时: 中位 {pct(im,50):.0f}ms  max {im.max():.0f}ms | 每 {per_cycle:.0f}ms 控制周期阻塞一次"
      f" → 冻结占空比 {im.mean()/per_cycle*100:.0f}%")

# ---- ⑥ 跟踪滞后(action−state) ----
lag = np.abs(A[:, ARM] - S[:, ARM]).max(axis=1)
print(f"[⑥] |命令-state| 臂维max: 中位 {pct(lag,50):.3f}  p90 {pct(lag,90):.3f}  max {lag.max():.3f}")

json.dump({"within_med": pct(within,50), "boundary_med": pct(boundary,50),
           "boundary_max": float(boundary.max()), "infer_med_ms": pct(im,50),
           "duty_pct": im.mean()/per_cycle*100, "wall_min": wall_min},
          open(OUT_DIR / "summary.json", "w"), indent=1)
print(f"\n[完成] summary.json + 全部 CSV → {OUT_DIR}")
