#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正版开局探针:FK 对齐(补腰) + 逐组/逐维超前 + 观测是否被原地改动"""
import sys, os
from pathlib import Path
_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _PY and Path(_PY).exists():
    os.execv(_PY, [_PY] + sys.argv)
import numpy as np, importlib.util, mujoco
# 用法: python home_chunk_probe2.py <hf_ckpt目录>; 不带参数 = w2@120000(基线)
_CKPT_ARG = sys.argv[1] if len(sys.argv) > 1 else None

PROJECT_ROOT = Path("/home/mjq/robot_item/lingbot-vla-v2-main")
sys.argv = [sys.argv[0]]; os.chdir(str(PROJECT_ROOT))
spec = importlib.util.spec_from_file_location("rollout_mod",
    PROJECT_ROOT / "mjq_lingbotvla_v2" / "Lingbot_H1_Rollout.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

model = mujoco.MjModel.from_xml_path(str(R.MJCF_PATH))
data = mujoco.MjData(model)
env = R.H1RolloutEnv(model, data, spawn_mode="home")
env.reset_episode(1.0)
images = env.render_images()
state0 = env.get_state()
OBS = {**images, "observation.state": state0, "task": R.DEFAULT_TASK}
state_backup = state0.copy()

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
QADR = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
        for j in R.UPPER_BODY_JOINTS]
HOME_WAIST = data.qpos[QADR[:2]].copy()
hand_ids = {s: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
               f"{s}_omnipicker_hand_{k}_loop_Link") for k in ("narrow", "wide")]
            for s in ("left", "right")}
def fk(row18):  # row18=[L7,R7,头2,爪2] → 补 home 腰 → 写 qpos
    data.qpos[QADR] = np.concatenate([HOME_WAIST, row18[:16]])
    mujoco.mj_forward(model, data)
    return (np.mean([data.xpos[i] for i in hand_ids["right"]], axis=0),
            np.mean([data.xpos[i] for i in hand_ids["left"]], axis=0))

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
CKPT = _CKPT_ARG or str(PROJECT_ROOT / "output/h1_v4_w2/checkpoints/global_step_120000/hf_ckpt")
print(f"[ckpt] {CKPT}\n")
policy = LingbotVLAv2Server(CKPT,
                            robot_norm_path=str(PROJECT_ROOT / "assets/norm_stats/h1_v4.json"),
                            use_length=50, chunk_ret=False, use_bf16=True, use_fp32=False, use_compile=False)
LM = {"左块": np.array([0.65, 0.3, 0.741]), "右块": np.array([0.65, -0.3, 0.741]), "桶": np.array([0.7, 0.0, 0.70])}
p0R, p0L = fk(state0[2:18])
print(f"home 指端 右{np.round(p0R,3)} 左{np.round(p0L,3)}(演示右发起段Δz≈−0.08 下移)\n")
print(f"{'采样':3s} {'右臂第一块Δ':26s} {'左臂Δ':24s} {'L超前':7s} {'R超前':7s} 右终点最近地标")
for k in range(5):
    policy.reset(R.ROBO_NAME)
    policy.infer(dict(OBS))
    ch = np.asarray(policy.last_action_chunk["action"][0], dtype=np.float32)  # (50,18)
    pR = np.array([fk(c)[0] for c in ch]); pL = np.array([fk(c)[1] for c in ch])
    dR, dL = pR[-1]-pR[0], pL[-1]-pL[0]
    lad = np.abs(ch[0,0:7]  - state0[2:9]).max()
    rad = np.abs(ch[0,7:14] - state0[9:16]).max()
    ds = {n: float(np.linalg.norm(pR[-1]-p)) for n,p in LM.items()}
    near = min(ds, key=ds.get)
    print(f"#{k+1:2d} [{dR[0]:+.3f},{dR[1]:+.3f},{dR[2]:+.3f}]      [{dL[0]:+.3f},{dL[1]:+.3f},{dL[2]:+.3f}]  {lad:6.3f} {rad:6.3f}  {near}({ds[near]:.2f})")
print(f"\n[防改动] 5 次推理后观测 state 是否被改: max|Δ| = {float(np.abs(state0-state_backup).max()):.2e}")
# 逐维细节(第1次采样)
policy.reset(R.ROBO_NAME); policy.infer(dict(OBS))
ch = np.asarray(policy.last_action_chunk["action"][0], dtype=np.float32)
ld = np.abs(ch[0] - state0[2:20])
print(f"\n[采样6 逐维超前18维] L臂{np.round(ld[0:7],2)} R臂{np.round(ld[7:14],2)} 头{np.round(ld[14:16],2)} 爪cmd{np.round(ch[0,16:18],2)}")
print(f"块内右臂总运动 {np.abs(np.diff(ch[:,7:14],axis=0)).sum():.2f} rad | 左臂 {np.abs(np.diff(ch[:,0:7],axis=0)).sum():.2f} rad")
