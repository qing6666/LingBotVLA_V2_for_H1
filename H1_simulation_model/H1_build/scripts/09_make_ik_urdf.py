#!/usr/bin/env python3
"""Step 9: 生成纯运动学 IK 用的 URDF —— 给遥操作 (Step 10) 用的 "轻装模型"。

为什么需要这个文件?
  MuJoCo 模型 (H1_scene.xml) 是给物理仿真用的: 带碰撞体、执行器、传感器。
  遥操作还需要另一个模型给 placo IK 用, 两者关注点不同:
    - IK 只需要 运动链 (joint 树) —— 不需要 visual/collision mesh (加载慢)
    - IK 需要一个 "TCP link" (工具中心点): 前辈把指尖中心表达成 link,
      我在 MuJoCo 里用的是 site (left/right_omnipicker_tcp), 数学上等价,
      但 URDF 侧必须真的有这个 link, placo 的 add_frame_task 才有目标可抓

输入: urdf/H1_fixed.urdf          (Step 2 产物, mesh 路径已修)
输出: urdf/H1_ik.urdf

处理三件事:
  ① 删掉每个 link 的 <visual> 和 <collision>   (纯运动学, 加载快且不依赖 STL)
  ② 追加左右 TCP link —— 变换抄自前辈 h1_for_import.urdf 的合并形式:
       Left_Link_J7  --xyz "0 0 -0.175", rpy "0 π 0"-->  left_omnipicker_tcp_link
       Right_Link_J7 --xyz "0 0 -0.175", rpy "π 0 0"-->  right_omnipicker_tcp_link
     (等价于 MJCF 里 mount -0.045 + 翻转 + site 0.13 的三段串联)
  ③ 自测: home 姿态下 placo FK 与 MuJoCo H1_scene.xml 的 TCP site 必须一致,
     再跑一次 mini-IK 证明这个 URDF 能被 KinematicsSolver 正常驱动

用法: python scripts/09_make_ik_urdf.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "urdf/H1_fixed.urdf"
DST = HERE / "urdf/H1_ik.urdf"
MJCF = HERE / "mujoco/H1_scene.xml"

PI = "3.141592653589793"
# TCP 变换: 父 link -> (xyz, rpy) -> TCP link。抄自前辈 h1_for_import.urdf
TCP_JOINTS = {
    "left_omnipicker_tcp_joint": {
        "parent": "Left_Link_J7",
        "child": "left_omnipicker_tcp_link",
        "xyz": "0 0 -0.175",
        "rpy": f"0 {PI} 0",
    },
    "right_omnipicker_tcp_joint": {
        "parent": "Right_Link_J7",
        "child": "right_omnipicker_tcp_link",
        "xyz": "0 0 -0.175",
        "rpy": f"{PI} 0 0",
    },
}


def make_ik_urdf() -> None:
    """① 删 visual/collision  ② 追加 TCP link  ③ 落盘。"""
    tree = ET.parse(SRC)
    root = tree.getroot()

    dropped = 0
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
                dropped += 1

    # fixed joint 无 <limit>, child link 无 inertial: pinocchio 会把 fixed joint
    # 合并进父 link, 惯量不受影响 —— 前辈的 h1_for_import.urdf 就是这么写的
    for name, spec in TCP_JOINTS.items():
        ET.SubElement(root, "link", {"name": spec["child"]})
        joint = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
        ET.SubElement(joint, "origin",
                      {"xyz": spec["xyz"], "rpy": spec["rpy"]})
        ET.SubElement(joint, "parent", {"link": spec["parent"]})
        ET.SubElement(joint, "child", {"link": spec["child"]})

    tree.write(DST, encoding="unicode", xml_declaration=True)
    print(f"✅ 已生成 {DST.relative_to(HERE)}  (删除 {dropped} 个 visual/collision, "
          f"追加 {len(TCP_JOINTS)} 个 TCP link)")


def read_home_from_mjcf() -> dict[str, float]:
    """从 H1_scene.xml 的 home keyframe 按关节名读姿态。

    教训 (Step 8 验证过): 长数字串绝不能手抄, 必须按名字映射。
    """
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        home[name] = float(model.key_qpos[key, model.jnt_qposadr[j]])
    return home


def self_test() -> None:
    """自测 A: 双模型 home FK 一致性; 自测 B: mini-IK 能收敛。"""
    import mujoco
    import numpy as np
    import placo

    # ── A1: MuJoCo 侧, home 下的 TCP site 世界坐标 ──────
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)

    # ── A2: placo 侧, 加载 IK 模型, 按名字灌入同样的关节角 ──
    robot = placo.RobotWrapper(str(DST))
    n_joints = len(robot.joint_names())
    assert n_joints == 18, f"关节数应为 18, 实际 {n_joints}: {list(robot.joint_names())}"
    home = read_home_from_mjcf()
    for name in robot.joint_names():
        assert name in home, f"URDF 关节 {name} 在 MJCF home 里找不到"
        robot.set_joint(name, home[name])
    robot.update_kinematics()

    # ── A3: 逐侧对比 (placo 基座 identity vs MuJoCo 根 body 世界位姿) ──
    print("── 自测 A: home 姿态 FK 一致性 ──")
    for side in ("left", "right"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                    f"{side}_omnipicker_tcp")
        mj_pos = data.site_xpos[site_id]
        pl_T = robot.get_T_world_frame(f"{side}_omnipicker_tcp_link")
        err = np.linalg.norm(mj_pos - pl_T[:3, 3])
        mark = "✅" if err < 1e-3 else "❌"
        print(f"  {mark} {side:5s} TCP  MuJoCo={np.round(mj_pos, 4)}  "
              f"placo={np.round(pl_T[:3, 3], 4)}  误差={err * 1e3:.3f} mm")
        assert err < 1e-3, f"{side} TCP 误差超 1mm, TCP 变换或基座对不上"

    # ── B: mini-IK —— Step 10 的核心循环提前验证 ──────
    # 配方来自前辈 h1_pico_teleop.py 的实测有效的用法:
    #   posture 任务钉住零空间 (没有它关节会漂到限位, 实测踩坑)
    #   每个控制周期只调一次 solve(True), 目标连续小步移动
    print("── 自测 B: mini-IK (左臂 TCP 前伸 10cm) ──")
    solver = robot.make_solver()
    solver.mask_fbase(True)
    solver.enable_joint_limits(True)
    posture = solver.add_joints_task()
    # home 是从 MJCF 读的全表 (含 34 个关节); IK 模型只有 18 个上身关节,
    # 夹爪关节不在 URDF 里, 必须过滤掉, 否则 set_joints 直接报错
    posture.set_joints({n: home[n] for n in robot.joint_names()})
    posture.configure("home_posture", "soft", 1e-3)

    T0 = robot.get_T_world_frame("left_omnipicker_tcp_link")
    target = T0.copy()
    target[0, 3] += 0.10
    task = solver.add_frame_task("left_omnipicker_tcp_link", T0)
    task.configure("left_tcp", "soft", 1.0)
    task.T_world_frame = target

    for _ in range(10):
        solver.solve(True)
        # 关键 (实测踩坑): solve(True) 只写 state.q, 不刷新运动学;
        # 下一轮的雅可比必须靠 update_kinematics() 更新,
        # 连续裸调 solve 会在旧线性化点上越走越偏 (实测发散到 879mm)
        robot.update_kinematics()
    reached = robot.get_T_world_frame("left_omnipicker_tcp_link")[:3, 3]
    err = np.linalg.norm(reached - target[:3, 3])
    print(f"  {'✅' if err < 5e-3 else '❌'} 10 轮后距目标 {err * 1e3:.2f} mm")
    assert err < 5e-3, "IK 不收敛, 检查任务配置"


def main() -> None:
    make_ik_urdf()
    self_test()
    print("\n=== Step 9 完成: H1_ik.urdf 就绪, 可以喂给 Step 10 键盘遥操作 ===")


if __name__ == "__main__":
    main()
