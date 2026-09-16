#!/usr/bin/env python3
"""Step 4: 加执行器 —— 给 18 个关节装上"虚拟电机", 让机器人能保持姿态。

输入: mujoco/H1.xml        (Step 3 产物: 无执行器, 会瘫软)
输出: mujoco/H1_actuated.xml

往 MJCF 里注入三个区块:
  <option>   仿真参数: 步长 1ms / 重力 / implicitfast 积分器(对接触更稳)
  <default>  按电机型号分 5 档默认参数(执行器 kp/kv + 关节 armature/frictionloss)
  <actuator> 每个关节一个 <position> 执行器, ctrlrange 直接复用关节限位

关键公式(MuJoCo position 执行器内置 PD 控制):
  力矩 = kp * (ctrl - q) - kv * qvel
  写 d.ctrl[i] = 目标弧度, 关节就会自己跟踪过去。

自测(脚本最后): 把所有关节设到行程中点作为目标, 仿真 3 秒,
最大漂移 < 0.05 rad 才算通过 —— 这就是"能站住"的量化标准。
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "mujoco/H1.xml"
OUT = HERE / "mujoco/H1_actuated.xml"

# ── 关节 -> 电机型号分档 (与 02_fix_urdf.py 的 SPEC 对应) ──────────────
def motor_class(name: str) -> str:
    if name == "waist_J2":
        return "hbn110_087"          # 腰 pitch, 最大电机 87Nm
    if name == "waist_J1":
        return "hbn90_052"           # 腰 yaw, 52Nm
    if name.endswith(("J1", "J2")) and name.startswith(("Left", "Right")):
        return "hbn80_031"           # 肩部 31Nm
    if name.endswith(("J3", "J4")) and name.startswith(("Left", "Right")):
        return "hbn70_010"           # 肘部 10Nm
    return "hbn52_005"               # 腕部 J5-J7 + 头部, 5Nm

# ── 每档的执行器/关节参数 (来自参考工程调好的值) ────────────────────────
# kp: 位置增益(拉向目标)  kv: 速度增益(刹车)  armature: 电机转子惯量(稳定仿真)
# frictionloss: 静摩擦(小抖动滤掉)  effort: 电机额定力矩(forcerange 钳位)
CLASS_PARAMS = {
    "hbn110_087": dict(kp=5500, kv=550, armature=0.12, frictionloss=0.25, effort=87),
    "hbn90_052":  dict(kp=4500, kv=450, armature=0.10, frictionloss=0.20, effort=52),
    "hbn80_031":  dict(kp=3500, kv=350, armature=0.06, frictionloss=0.12, effort=31),
    "hbn70_010":  dict(kp=2000, kv=200, armature=0.025, frictionloss=0.06, effort=10),
    "hbn52_005":  dict(kp=600, kv=60, armature=0.01, frictionloss=0.03, effort=5),
}


def build() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()

    # ① <option>: 仿真参数 (插在 compiler 后面)
    #    tolerance 收紧到 1e-10: 约束求解更精确 (夹爪耦合对手指一致性敏感)
    #    cone=elliptic + noslip_iterations=10 (2026-08-19 对齐参考工程):
    #      · 椭圆摩擦锥 + condim=6 的滚转/扭转摩擦才准, 金字塔锥是线性近似;
    #      · noslip 后处理消除接触面切向微滑 —— 没有它方块会在指间蠕滑,
    #        抓取发"酥", 这是"夹取有阻力夹不上"手感的来源之一。
    root.insert(1, ET.Element("option", {
        "timestep": "0.001", "gravity": "0 0 -9.81",
        "integrator": "implicitfast", "iterations": "100", "tolerance": "1e-10",
        "cone": "elliptic", "noslip_iterations": "10",
    }))

    # ② <default>: 按电机型号分档的默认参数 (插在 option 后面)
    default = ET.Element("default")
    for cls, p in CLASS_PARAMS.items():
        sub = ET.SubElement(default, "default", {"class": cls})
        ET.SubElement(sub, "position", {"kp": str(p["kp"]), "kv": str(p["kv"])})
        ET.SubElement(sub, "joint", {
            "armature": str(p["armature"]),
            "frictionloss": str(p["frictionloss"]),
        })
    root.insert(2, default)

    # ③ 给每个关节挂档 + 生成执行器
    actuator = ET.Element("actuator")
    for joint in root.findall(".//worldbody//joint"):
        name = joint.get("name")
        cls = motor_class(name)
        joint.set("class", cls)                       # 关节继承该档的 armature 等
        # 坑: URDF <dynamics friction> 转 MJCF 时写死了 frictionloss=0.1,
        # 元素自带属性会压过 class 默认值 —— 必须删掉, 分档值才能生效!
        joint.attrib.pop("frictionloss", None)
        effort = CLASS_PARAMS[cls]["effort"]
        control = {"name": name + "_position", "joint": name, "class": cls,
                   "forcerange": f"-{effort} {effort}"}   # 出力不超过电机额定
        if joint.get("range"):                        # ctrlrange = 关节限位
            control["ctrlrange"] = joint.get("range")
        actuator.append(ET.Element("position", control))
    root.append(actuator)                             # actuator 放在文件末尾区

    ET.indent(tree, space="  ")
    OUT.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    print(f"✅ 已生成 {OUT}")
    print(f"   执行器: {len(actuator)} 个 (期望 18)")
    print(f"   电机分档: {', '.join(f'{c}={sum(1 for j in root.findall(".//worldbody//joint") if motor_class(j.get("name"))==c)}关节' for c in CLASS_PARAMS)}")


def self_test() -> None:
    """稳定性自测: 目标=行程中点, 保持 3 秒, 漂移要小。"""
    model = mujoco.MjModel.from_xml_path(str(OUT))
    data = mujoco.MjData(model)

    # 目标姿态: 每个关节取行程中点, 写进 ctrl (执行器目标)
    target = np.zeros(model.nq)
    for j in range(model.njnt):
        lo, hi = model.jnt_range[j]
        target[model.jnt_qposadr[j]] = 0.5 * (lo + hi)
    data.ctrl[:] = target[:model.nu]

    # 先仿真 3 秒让它收敛到目标, 再测之后 3 秒的漂移
    for _ in range(3000):
        mujoco.mj_step(model, data)
    q_ref = data.qpos.copy()
    for _ in range(3000):
        mujoco.mj_step(model, data)
    drift = float(np.abs(data.qpos - q_ref).max())

    print(f"\n── 稳定性自测 ──")
    print(f"   执行器数 nu={model.nu} | 目标=行程中点 | 6秒仿真")
    print(f"   最大漂移: {drift:.5f} rad", "✅ 通过(<0.05)" if drift < 0.05 else "❌ 不稳定")


if __name__ == "__main__":
    build()
    self_test()
