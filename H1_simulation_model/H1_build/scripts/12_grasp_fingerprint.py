#!/usr/bin/env python3
"""Step 12: 抓取接触几何指纹 —— 与参考工程逐点对照手指碰撞体几何。

背景 (2026-08-19 穿模+弹飞事故的复盘):
  手指碰撞体曾用 48% AABB 盒, 比真实手指平均胖 6.35mm —— 方块离真手指
  还有 6mm 时盒角已经开始硬推它, 表现为"穿模"(看着没碰到其实在碰)和
  "弹飞"(硬接触参数下盒角的法向冲量)。修复 = 手指改用 STL 凸包碰撞体
  (type="mesh", Step 6), 与参考工程一致。

本脚本验证"修好了且没修歪": 在 开度×方块位置 网格上, 用 mj_geomDistance
测方块到每根手指碰撞体的精确距离, 与参考工程逐点对比:
  平均偏差 < 0.5 mm 且 90% 以上点位 < ±1 mm  → ✅

  (不用动力学做判据: "瞬移方块进指间再闭合"测的是求解器分离初始重叠的
   烈度, 不是抓取质量 —— 连参考工程都会把深重叠的方块弹飞 6 m/s。)

用法:  python scripts/12_grasp_fingerprint.py [--reference 参考xml]
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent.parent
SCENE = HERE / "mujoco/H1_scene.xml"
DEFAULT_REF = HERE.parent / "H1_1" / "mujoco" / "h1_omnipicker.xml"

CTRLS = (1.0, 0.75, 0.5)          # 夹爪开度采样
EXTRAS = np.arange(-0.02, 0.061, 0.004)   # 方块沿指轴的位置采样
SIDE = "right"
CUBE = "red_cube_right"


def fingerprint(xml: Path) -> dict:
    """{(ctrl, extra, geom_id): 方块到该手指碰撞体的距离} —— 接触几何指纹。"""
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)

    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                             f"{SIDE}_omnipicker_gripper_opening")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE)
    jadr = model.jnt_qposadr[model.body_jntadr[bid]]
    dadr = model.jnt_dofadr[model.body_jntadr[bid]]
    cube_geom = int(model.body_geomadr[bid])

    hands = sorted(
        g for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        .startswith(SIDE) and "_hand_" in
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))

    data.ctrl[grip] = 1.0                       # 先全开 (蠕变需要时间)
    for _ in range(2500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    center = np.mean([data.geom_xpos[g] for g in hands], axis=0)
    z = data.site_xmat[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE,
        f"{SIDE}_omnipicker_tcp")].reshape(3, 3)[:, 2]

    table = {}
    for ctrl in CTRLS:
        data.ctrl[grip] = ctrl
        for _ in range(1200):                   # 1.2s 蠕变到该开度
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        for extra in EXTRAS:
            data.qpos[jadr:jadr + 3] = center + z * extra
            data.qpos[jadr + 3:jadr + 7] = (1, 0, 0, 0)
            data.qvel[dadr:dadr + 6] = 0
            mujoco.mj_forward(model, data)
            for g in hands:
                table[(round(ctrl, 2), round(float(extra), 3), g)] = \
                    mujoco.mj_geomDistance(model, data, cube_geom, int(g),
                                            5.0, np.zeros(6))
    return table


def count_mesh_fingers(xml: Path) -> int:
    root = ET.parse(xml).getroot()
    return sum(1 for g in root.iter("geom")
               if g.get("type") == "mesh"
               and g.get("class") == "finger_collision")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REF)
    args = parser.parse_args()

    n_finger = count_mesh_fingers(SCENE)
    print(f"[指纹] 手指 mesh 碰撞体: {n_finger} 个 (期望 16 = 8 种网格 × 左右手)")
    if n_finger != 16:
        print("[指纹] ❌ 数量不对 —— 手指碰撞体不是 mesh? 先跑 06_collision.py")
        return 1

    print(f"[指纹] 采样网格: 开度 {list(CTRLS)} × 方块位置 {len(EXTRAS)} 点"
          f" × 手指 8 件 (约 {len(CTRLS)*len(EXTRAS)*8} 距离/模型)")
    ours = fingerprint(SCENE)
    ref = fingerprint(args.reference)
    keys = sorted(ref.keys())

    diffs = np.array([ours[k] - ref[k] for k in keys])
    near = float((np.abs(diffs) < 0.001).mean())
    print(f"\n[指纹] vs 参考工程 {args.reference.name}:")
    print(f"  平均偏差   {diffs.mean()*1000:+.2f} mm   (阈值 ±0.5)")
    print(f"  |偏差|<1mm 占比 {near*100:.0f}%   (阈值 90%)")
    ok = abs(diffs.mean()) < 0.0005 and near > 0.90
    print(f"  判定: {'✅ 接触几何与参考工程等价' if ok else '❌ 几何有偏差'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
