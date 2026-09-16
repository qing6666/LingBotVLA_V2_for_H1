#!/usr/bin/env python3
"""Step 8: 对照参考工程 —— 我们从零搭的模型和前辈的最终版差在哪?

输入: mujoco/H1_scene.xml                       (Step 7 产物)
      H1_1/mujoco/h1_omnipicker.xml              (参考工程最终版)
输出: 终端对照报告 (✅ 一致 / ⚠️ 有差异, 按影响排序)

对比维度:
  ① 规模:   nq/nu/nbody/nmesh/ngeom/nsensor/neq/nkey/ncamera
  ② 关节:   限位/阻尼/转子惯量/静摩擦 (按名字逐个比)
  ③ 执行器: kp/kv/力限幅/控制限幅
  ④ 约束:   数量/类型/求解参数
  ⑤ 传感器: 谁有谁没有
  ⑥ home:   keyframe 逐关节姿态差
  ⑦ 求解器: option 参数

用法: python scripts/08_compare.py
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent.parent
MINE = HERE / "mujoco/H1_scene.xml"
EXPERT = Path("/home/mjq/robot_item/lerobot-main/H1_simulation_model/H1_1"
              "/mujoco/h1_omnipicker.xml")


def joints_by_name(m):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j
            for j in range(m.njnt)}


def actuator_row(m, a):
    g = m.actuator_gainprm[a]
    b = m.actuator_biasprm[a]
    return (f"kp={g[0]:.0f} kv={-b[2]:.0f} "
            f"force=[{m.actuator_forcerange[a][0]:.1f},{m.actuator_forcerange[a][1]:.1f}] "
            f"ctrl=[{m.actuator_ctrlrange[a][0]:.2f},{m.actuator_ctrlrange[a][1]:.2f}]")


def main() -> None:
    mine = mujoco.MjModel.from_xml_path(str(MINE))
    exp = mujoco.MjModel.from_xml_path(str(EXPERT))

    # ① 规模
    print("① 规模")
    for label, attr in [("nq", "nq"), ("nu", "nu"), ("nbody", "nbody"),
                        ("nmesh", "nmesh"), ("ngeom", "ngeom"),
                        ("nsensor", "nsensor"), ("neq", "neq"),
                        ("nkey", "nkey"), ("ncam", "ncam")]:
        a, b = getattr(mine, attr), getattr(exp, attr)
        mark = "✅" if a == b else "⚠️ "
        print(f"   {mark} {label:8s} 我={a:<4d} 前辈={b}")

    # ② 关节逐项
    print("\n② 关节 (限位/阻尼/转子惯量/静摩擦)")
    mj, ej = joints_by_name(mine), joints_by_name(exp)
    only_exp = set(ej) - set(mj)
    only_mine = set(mj) - set(ej)
    if only_exp:
        print(f"   ⚠️  前辈多出: {sorted(only_exp)}")
    if only_mine:
        print(f"   ⚠️  我多出: {sorted(only_mine)}")
    n_diff = 0
    for name, j in sorted(mj.items()):
        if name not in ej:
            continue
        k = ej[name]
        da, db = mine.jnt_dofadr[j], exp.jnt_dofadr[k]
        rows = []
        if not np.allclose(mine.jnt_range[j], exp.jnt_range[k], atol=1e-3):
            rows.append(f"限位 {mine.jnt_range[j]} vs {exp.jnt_range[k]}")
        for attr, label in [("dof_damping", "阻尼"), ("dof_armature", "转子"),
                            ("dof_frictionloss", "摩擦")]:
            va, vb = getattr(mine, attr)[da], getattr(exp, attr)[db]
            if abs(va - vb) > 1e-6:
                rows.append(f"{label} {va:.3g} vs {vb:.3g}")
        if rows:
            n_diff += 1
            print(f"   ⚠️  {name}: " + "; ".join(rows))
    if n_diff == 0:
        print("   ✅ 全部一致")

    # ③ 执行器逐项
    print("\n③ 执行器")
    n_diff = 0
    for a in range(mine.nu):
        name = mujoco.mj_id2name(mine, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        k = mujoco.mj_name2id(exp, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if k < 0:
            print(f"   ⚠️  前辈没有: {name}")
            continue
        ra, rb = actuator_row(mine, a), actuator_row(exp, k)
        if ra != rb:
            n_diff += 1
            print(f"   ⚠️  {name}\n        我    {ra}\n        前辈 {rb}")
    if n_diff == 0:
        print("   ✅ 全部一致")

    # ④ 约束
    print("\n④ 约束")
    types = {int(t): 0 for t in set(mine.eq_type) | set(exp.eq_type)}
    for t in mine.eq_type:
        types[int(t)] += 1
    etypes = {int(t): 0 for t in exp.eq_type}
    for t in exp.eq_type:
        etypes[int(t)] += 1
    name_of = {0: "connect", 2: "joint"}
    same = all(types.get(t, 0) == etypes.get(t, 0) for t in set(types) | set(etypes))
    detail = ", ".join(f"{name_of.get(t, t)}×{types.get(t, 0)}/{etypes.get(t, 0)}"
                       for t in sorted(set(types) | set(etypes)))
    print(f"   {'✅' if same else '⚠️ '} 类型计数 (我/前辈): {detail}")
    sol_mine = {tuple(np.round(mine.eq_solref[e], 6)) for e in range(mine.neq)}
    sol_exp = {tuple(np.round(exp.eq_solref[e], 6)) for e in range(exp.neq)}
    print(f"   {'✅' if sol_mine == sol_exp else '⚠️ '} solref: 我={sol_mine} 前辈={sol_exp}")

    # ⑤ 传感器
    print("\n⑤ 传感器")
    exp_sensors = {}
    for s in range(exp.nsensor):
        stype = int(exp.sensor_type[s])
        exp_sensors.setdefault(stype, []).append(
            mujoco.mj_id2name(exp, mujoco.mjtObj.mjOBJ_SENSOR, s))
    name_of = {int(mujoco.mjtSensor.mjSENS_TOUCH): "touch",
               int(mujoco.mjtSensor.mjSENS_FRAMEPOS): "framepos",
               int(mujoco.mjtSensor.mjSENS_FRAMEQUAT): "framequat"}
    summary = ", ".join(f"{name_of.get(t, t)}×{len(v)}"
                        for t, v in sorted(exp_sensors.items()))
    mark = "✅" if mine.nsensor == exp.nsensor else "⚠️ "
    print(f"   {mark} 数量 我={mine.nsensor} 前辈={exp.nsensor} "
          f"(构成: {summary})")

    # ⑥ home keyframe
    print("\n⑥ home 姿态")
    kh_m = mujoco.mj_name2id(mine, mujoco.mjtObj.mjOBJ_KEY, "home")
    kh_e = mujoco.mj_name2id(exp, mujoco.mjtObj.mjOBJ_KEY, "home")
    bad = []
    for name, j in mj.items():
        if name not in ej:
            continue
        qa = mine.key_qpos[kh_m][mine.jnt_qposadr[j]]
        qb = exp.key_qpos[kh_e][exp.jnt_qposadr[ej[name]]]
        if abs(qa - qb) > 1e-4:
            bad.append(f"{name}: {qa:+.4f} vs {qb:+.4f}")
    if bad:
        print("   ⚠️  " + "; ".join(bad))
    else:
        print("   ✅ 逐关节一致 (含 freejoint 方块位姿)")

    # ⑦ 求解器
    print("\n⑦ 求解器 option")
    for label, va, vb in [
        ("timestep", mine.opt.timestep, exp.opt.timestep),
        ("integrator", int(mine.opt.integrator), int(exp.opt.integrator)),
        ("iterations", mine.opt.iterations, exp.opt.iterations),
        ("tolerance", mine.opt.tolerance, exp.opt.tolerance),
    ]:
        mark = "✅" if va == vb else "⚠️ "
        print(f"   {mark} {label:12s} 我={va} 前辈={vb}")


if __name__ == "__main__":
    main()
