#!/usr/bin/env python3
"""Step 5: 装上双 OmniPicker 夹爪 —— 学习"部件合并"的完整套路。

输入: mujoco/H1_actuated.xml          (Step 4 产物: 18 关节 + 18 执行器)
      H1_1/mujoco/omnipicker_component.xml  (夹爪 MJCF 组件, "采购件")
输出: mujoco/H1_grippers.xml          (期望 nq=34, nu=20)

合并一台机器人 + 两个相同末端部件, 要做对四件事:

  ① 挂载变换: H1 手腕沿 -Z 伸出, 夹爪沿 +Z 安装 → 翻转 180°;
     左手绕 Y 翻(rpy=0,π,0), 右手绕 X 翻(rpy=π,0,0), 让宽手指朝外侧。
  ② 命名前缀: 左右所有 link/joint/equality/actuator 名加
     left_omnipicker_ / right_omnipicker_ 前缀, 防止撞名。程序化改, 不手改。
  ③ 机械耦合: 组件里的 <equality>(四连杆 connect + 相向耦合 joint)
     原样复制 —— 8 个手指关节共享 1 个开合执行器的"机械变速箱"。
  ④ 资源去重: 两侧共用同一套 mesh 资产(omnipicker_ 前缀只加一次)。

自测: 夹爪开合指令 0→1→0, 手指关节必须跟着动; 整机仿真不发散。
"""

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "mujoco/H1_actuated.xml"
OUT = HERE / "mujoco/H1_grippers.xml"
GRIPPER = Path("/home/mjq/robot_item/lerobot-main/H1_simulation_model/H1_1"
               "/mujoco/omnipicker_component.xml")          # "采购件"
GRIPPER_MESHES = GRIPPER.parent.parent / "meshes/omnipicker"
MY_MESHES = HERE / "meshes/omnipicker"

# 安装参数 (与参考工程一致): 腕部 J7 沿 -Z 偏移 45mm + 180°翻转
MOUNTS = {
    "left":  ("Left_Link_J7",  "0 0 -0.045", "0 3.14159265 0"),
    "right": ("Right_Link_J7", "0 0 -0.045", "3.14159265 0 0"),
}

# 手指关节真实行程 (组件文件里是 ±3.14 占位值!) —— 参考工程实测收紧。
# 不收紧的下场: IK/遥操作可能给手指下发物理上不可能的角度。
FINGER_RANGES = {
    "hand_narrow1_joint":      "-1.25 0.05",
    "hand_narrow2_joint":      "-0.05 0.38",
    "hand_narrow3_joint":      "-0.22 0.08",
    "hand_narrow_loop_joint":  "-1.65 0.05",
    "hand_wide1_joint":        "-0.05 1.25",
    "hand_wide2_joint":        "-0.38 0.05",
    "hand_wide3_joint":        "-0.08 0.22",
    "hand_wide_loop_joint":    "-0.05 1.65",
}


def prefix_names(node: ET.Element, prefix: str) -> ET.Element:
    """深拷贝一棵 XML 子树, 给所有名字类属性加前缀。"""
    result = copy.deepcopy(node)
    for el in result.iter():
        orig_name = el.get("name")
        if orig_name:
            el.set("name", prefix + orig_name)
        for attr in ("body1", "body2", "joint", "joint1", "joint2"):  # 约束引用
            if el.get(attr):
                el.set(attr, prefix + el.get(attr))
        if el.tag == "geom" and el.get("mesh"):
            el.set("mesh", "omnipicker_" + el.get("mesh"))            # 共享 mesh 资产
        elif el.tag == "joint":
            el.set("class", "omnipicker_joint")                        # 用夹爪档参数
            if orig_name in FINGER_RANGES:
                el.set("range", FINGER_RANGES[orig_name])             # 占位值→真实行程
    return result


def add_one_gripper(root, gripper_root, side: str) -> None:
    """把一只夹爪装到对应手腕上。"""
    parent_name, pos, euler = MOUNTS[side]
    prefix = f"{side}_omnipicker_"

    # ①② 挂载点: 在腕部 body 下创建 mount body (含安装座惯量 + 两个测量 site)
    wrist = root.find(f".//body[@name='{parent_name}']")
    if wrist is None:
        raise ValueError(f"找不到腕部 body: {parent_name}")
    mount = ET.SubElement(wrist, "body", {
        "name": prefix + "mount", "pos": pos, "euler": euler,
    })
    # 安装座惯量 (参考工程实测值: 256g)
    ET.SubElement(mount, "inertial", {
        "pos": "-0.0000552 0.000012341 0.03296193", "mass": "0.25641368",
        "fullinertia": "0.00046351 0.00044525 0.00010438 -1e-8 -1.04e-6 -5e-8",
    })
    # TCP site: 抓取中心 (夹爪基座前方 130mm), 以后 IK/传感器都挂它
    ET.SubElement(mount, "site", {"name": prefix + "tcp", "pos": "0 0 0.13",
                                  "size": "0.004"})
    ET.SubElement(mount, "site", {"name": prefix + "wrist_ft_site",
                                  "pos": "0 0 0", "size": "0.006"})

    # 夹爪本体: 组件 worldbody 的 geom/body 挂进 mount
    for el in gripper_root.find("worldbody"):
        if el.tag in ("geom", "body"):
            mount.append(prefix_names(el, prefix))

    # ③ 耦合约束: connect(四连杆) + joint(相向手指)
    #    solref 0.005→0.002 + solimp 补硬: 参考工程把约束调得更硬, 手指
    #    耦合响应更快更跟手。2026-08-19 复盘"夹取有阻力夹不上": 只改
    #    solref 不够, solimp 留默认(0.9 0.95 0.001)时四连杆是软的 ——
    #    手指一压到方块, 软约束被接触力顶开, 指尖回弹, 感觉夹不拢。
    equality = root.find("equality")
    for c in gripper_root.find("equality"):
        copied = prefix_names(c, prefix)
        copied.set("solref", "0.002 1")
        copied.set("solimp", "0.99 0.999 0.0001")
        equality.append(copied)

    # 执行器: 每侧 1 个开合执行器。组件出厂 kp=20 偏弱, 参考工程调到 60。
    # 教训: forcerange 是"开合速度"的隐形旋钮 —— 曾试过 ±0.8, 张开要 2 秒
    # (0.5s 时只到 -0.43, 误判成卡死); ±1.2 时 1 秒到位。限幅力要先于阻尼平衡。
    actuator = root.find("actuator")
    for drive in gripper_root.find("actuator"):
        copied = prefix_names(drive, prefix)
        copied.set("kp", "60")
        copied.set("kv", "4")
        copied.set("forcerange", "-1.2 1.2")
        actuator.append(copied)


def build() -> None:
    # ⓪ 复制夹爪 mesh 到我们工程
    MY_MESHES.mkdir(parents=True, exist_ok=True)
    for f in GRIPPER_MESHES.iterdir():
        shutil.copy(f, MY_MESHES / f.name)

    tree = ET.parse(SRC)
    root = tree.getroot()
    gripper_root = ET.parse(GRIPPER).getroot()

    # 给 defaults 补一档夹爪参数 (手指关节轻, 阻尼小)
    default = root.find("default")
    gsub = ET.SubElement(default, "default", {"class": "omnipicker_joint"})
    ET.SubElement(gsub, "joint", {"damping": "0.1", "armature": "0.001",
                                  "frictionloss": "0.01"})
    ET.SubElement(gsub, "position", {"kp": "80", "kv": "4"})

    # mesh 资产: 只加一份, 两侧共享
    asset = root.find("asset")
    for mesh in gripper_root.find("asset"):
        if mesh.tag != "mesh":
            continue
        copied = copy.deepcopy(mesh)
        copied.set("name", "omnipicker_" + mesh.get("name"))
        copied.set("file", "../meshes/omnipicker/" +
                   Path(mesh.get("file")).name)
        asset.append(copied)

    # equality 区块要放在 actuator 前面 (MJCF 惯例顺序)
    actuator = root.find("actuator")
    equality = ET.Element("equality")
    root.insert(list(root).index(actuator), equality)

    for side in ("left", "right"):
        add_one_gripper(root, gripper_root, side)

    ET.indent(tree, space="  ")
    OUT.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    print(f"✅ 已生成 {OUT}")


def self_test() -> None:
    model = mujoco.MjModel.from_xml_path(str(OUT))
    data = mujoco.MjData(model)

    print(f"\n── 自测 ──")
    print(f"   nq={model.nq} (期望 34 = 18本体 + 16手指)   nu={model.nu} (期望 20)")
    assert model.nq == 34 and model.nu == 20, "自由度/执行器数不对!"

    # 找到左右夹爪执行器和一根手指关节
    for side in ("left", "right"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"{side}_omnipicker_gripper_opening")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{side}_omnipicker_hand_narrow1_joint")
        assert aid >= 0 and jid >= 0, f"{side} 夹爪名字找不到"

        # 开合测试: ctrl 0 → 1 → 0, 手指角度必须跟着变
        q_before = data.qpos[model.jnt_qposadr[jid]].copy()
        data.ctrl[aid] = 1.0
        for _ in range(500):                       # 0.5 秒收敛
            mujoco.mj_step(model, data)
        q_open = data.qpos[model.jnt_qposadr[jid]].copy()
        data.ctrl[aid] = 0.0
        for _ in range(500):
            mujoco.mj_step(model, data)
        q_close = data.qpos[model.jnt_qposadr[jid]].copy()
        moved = abs(q_open - q_close) > 0.3        # 手指明显动了
        print(f"   {side:5s}夹爪: 张开{q_open:+.2f} → 闭合{q_close:+.2f} rad "
              f"{'✅ 联动正常' if moved else '❌ 手指没动'}")

    # 整机稳定性: 再跑 2 秒不发散
    for _ in range(2000):
        mujoco.mj_step(model, data)
    ok = np.isfinite(data.qpos).all()
    print(f"   整机 2s 仿真: {'✅ 无发散' if ok else '❌ 数值发散'}")


if __name__ == "__main__":
    build()
    self_test()
