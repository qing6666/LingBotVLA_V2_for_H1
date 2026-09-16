#!/usr/bin/env python3
"""Step 7: 搭场景 —— 把"机器人"变成"可以开始干活的工位"。

输入: mujoco/H1_collision.xml   (Step 6 产物: 只有机器人悬空站着)
输出: mujoco/H1_scene.xml       (期望 nq=48: 34 机器人 + 2 方块 × 7 自由体)

一个可用的桌面操作场景需要五样东西:

  ① 环境: 灯光 + 无限地板 + 工作桌(桌面 z=0.641) + 绿色料箱 + 两个红方块
     —— 方块带 freejoint(6 自由度刚体), 是被操作的对象。
  ② 碰撞规则闭环: 环境 geom contype=1 conaffinity=1,
     机器人 collision geom contype=0 conaffinity=1
     → 机器人碰环境 ✓ / 环境碰环境 ✓ / 机器人自碰 ✗ (Step 6 的设计)
  ③ 相机: 头部 head_rgb + 左右腕 wrist_rgb, 用 xyaxes 指定朝向。
     MuJoCo 相机沿本地 -Z 看; xyaxes 给相机坐标系的 X/Y 轴。
     以后 lerobot 采集数据就靠这三路"眼睛"。
  ④ home keyframe: 启动/回零的基准姿态 (双肘 J4=90°, 其余 0)。
     qpos/ctrl 一长串数字, 永远用"按名字映射"生成, 不手数索引!
  ⑤ 传感器: 142 路 (关节 pos/vel、力矩、触觉、腕力、TCP 位姿),
     以后 lerobot 采集数据的观测向量。
  ⑥ 自测: 方块落到桌上稳住 + 机器人保持 home 姿态不倒。

home 数值直接从参考工程的 keyframe 里按 joint/actuator 名字搬过来 ——
这也验证了我们前面 6 步的命名体系与参考工程完全对齐。
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "mujoco/H1_collision.xml"
OUT = HERE / "mujoco/H1_scene.xml"
EXPERT = Path("/home/mjq/robot_item/lerobot-main/H1_simulation_model/H1_1"
              "/mujoco/h1_omnipicker.xml")

# ── 三个相机 (与参考工程一致; xyaxes = 相机系 X轴 Y轴, 视线 = X×Y) ──────
CAMERAS = {
    "Head_Link_j2": dict(name="head_rgb", pos="0.065 0 0.035",
                         xyaxes="0 -1 0 0 0 1", fovy="75"),
    "left_omnipicker_mount": dict(name="left_wrist_rgb", pos="-0.06 0 0.055",
                                  xyaxes="0 -1 0 -0.94089 0 0.33872", fovy="100"),
    "right_omnipicker_mount": dict(name="right_wrist_rgb", pos="0.06 0 0.055",
                                   xyaxes="0 1 0 0.94089 0 0.33872", fovy="100"),
}


def env_geom(name: str, pos: str, size: str, rgba: str,
             friction: str = "0.9 0.02 0.001") -> ET.Element:
    """环境几何体: 桌/箱/方块共用 contype=1 conaffinity=1 的碰撞属性。"""
    return ET.Element("geom", {
        "name": name, "type": "box", "pos": pos, "size": size, "rgba": rgba,
        "contype": "1", "conaffinity": "1", "friction": friction, "condim": "3",
    })


def add_sensors(root: ET.Element) -> None:
    """给模型装上"神经系统" —— 以后 lerobot 采集数据的观测向量全靠它们。

    共 142 路, 与参考工程一致:
      jointpos/jointvel ×34      每个关节的角度和角速度 (状态观测)
      actuatorfrc ×20            每个执行器的输出力矩
      jointactuatorfrc ×34       每个关节(自由度)受到的总驱动力矩
      touch ×12                  指尖触觉 (Step 6 埋的 touch_site 在这里兑现!)
      force/torque ×4            腕部六维力传感器 (Step 5 埋的 ft_site)
      framepos/framequat ×4      TCP 世界坐标位姿 (抓取目标跟踪)
    """
    sensor = ET.Element("sensor")

    for joint in root.findall(".//worldbody//joint"):
        name = joint.get("name")
        ET.SubElement(sensor, "jointpos", {"name": name + "_pos", "joint": name})
        ET.SubElement(sensor, "jointvel", {"name": name + "_vel", "joint": name})
        ET.SubElement(sensor, "jointactuatorfrc",
                      {"name": name + "_actuator_torque", "joint": name})

    for act in root.find("actuator"):
        name = act.get("name")
        ET.SubElement(sensor, "actuatorfrc",
                      {"name": name + "_torque", "actuator": name})

    for site in root.findall(".//site"):
        name = site.get("name", "")
        if name.endswith("_touch_site"):
            ET.SubElement(sensor, "touch",
                          {"name": name[:-5], "site": name})
        elif name.endswith("wrist_ft_site"):
            ET.SubElement(sensor, "force", {"name": name + "_force", "site": name})
            ET.SubElement(sensor, "torque", {"name": name + "_torque", "site": name})
        elif name.endswith("_tcp"):
            ET.SubElement(sensor, "framepos",
                          {"name": name + "_pos", "objtype": "site", "objname": name})
            ET.SubElement(sensor, "framequat",
                          {"name": name + "_quat", "objtype": "site", "objname": name})

    root.insert(list(root).index(root.find("actuator")) + 1, sensor)


def build() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    # ①a 资产: 棋盘格地面材质 + 天空盒
    asset = root.find("asset")
    asset.append(ET.fromstring(
        '<texture name="groundplane" type="2d" builtin="checker" mark="edge"'
        ' rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"'
        ' width="300" height="300"/>'))
    asset.append(ET.fromstring(
        '<material name="groundplane" texture="groundplane" texrepeat="5 5"'
        ' texuniform="true" reflectance="0.2"/>'))
    asset.append(ET.fromstring(
        '<texture name="skybox" type="skybox" builtin="gradient"'
        ' rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>'))

    # ①b 地板档默认参数
    root.find("default").append(ET.fromstring(
        '<default class="floor"><geom type="plane" contype="1" conaffinity="1"'
        ' friction="1.0 0.02 0.001" condim="3" material="groundplane"/></default>'))

    # ①c 环境: 灯 / 地板 / 桌 / 箱 / 方块
    worldbody.append(ET.fromstring(
        '<light name="key_light" pos="0 0 2" dir="0 0 -1" directional="true"/>'))
    worldbody.append(ET.fromstring(
        '<geom name="floor" class="floor" pos="0 0 0" size="0 0 0.05"/>'))

    table = ET.SubElement(worldbody, "body", {"name": "work_table", "pos": "0.8 0 0"})
    table.append(env_geom("tabletop", "0 0 0.621", "0.5 0.5 0.02", "0.55 0.32 0.16 1"))
    for leg, (x, y) in {"front_left": (-0.4, 0.4), "front_right": (-0.4, -0.4),
                        "back_left": (0.4, 0.4), "back_right": (0.4, -0.4)}.items():
        table.append(env_geom(f"table_leg_{leg}", f"{x} {y} 0.3005",
                              "0.035 0.035 0.3005", "0.22 0.22 0.24 1",
                              friction="0.8 0.02 0.001"))

    # 绿色料箱: 底 + 四壁 (开口向上, 外边长 0.1m, 中心在机器人前方 0.7m)
    bin_body = ET.SubElement(worldbody, "body", {"name": "green_bin", "pos": "0.7 0 0"})
    bin_body.append(env_geom("green_bin_bottom", "0 0 0.6435", "0.05 0.05 0.0025",
                             "0.1 0.75 0.2 1"))
    for wall, (pos, size) in {
        "front": ("-0.0475 0 0.691", "0.0025 0.05 0.05"),
        "back":  ("0.0475 0 0.691", "0.0025 0.05 0.05"),
        "left":  ("0 0.0475 0.691", "0.05 0.0025 0.05"),
        "right": ("0 -0.0475 0.691", "0.05 0.0025 0.05"),
    }.items():
        bin_body.append(env_geom(f"green_bin_wall_{wall}", pos, size, "0.1 0.75 0.2 1"))

    # 红方块 ×2: freejoint 刚体, 密度 500 (边长 4cm → 32g)
    for side, y in (("left", 0.3), ("right", -0.3)):
        cube = ET.SubElement(worldbody, "body",
                             {"name": f"red_cube_{side}", "pos": f"0.25 {y} 0.741"})
        ET.SubElement(cube, "freejoint", {"name": f"red_cube_{side}_freejoint"})
        cube.append(ET.Element("geom", {
            "name": f"red_cube_{side}_geom", "type": "box",
            "size": "0.02 0.02 0.02", "rgba": "0.9 0.05 0.05 1", "density": "500",
            "contype": "1", "conaffinity": "1", "friction": "1.0 0.02 0.001",
            "condim": "4",
        }))

    # ③ 相机: 挂到对应 body 上
    for parent, cam in CAMERAS.items():
        host = root.find(f".//body[@name='{parent}']")
        if host is None:
            raise ValueError(f"相机宿主 body 不存在: {parent}")
        ET.SubElement(host, "camera", cam)

    # ⑤ 传感器套件 (142 路, 数据采集的观测向量)
    add_sensors(root)

    # ④ home keyframe 要知道"我们模型"的 qpos 布局才能按名字填数值,
    #    而 MJCF 编译又要求 mesh 文件在磁盘上 —— 所以先落盘, 编译, 再回填:
    ET.indent(tree, space="  ")
    OUT.write_text('<?xml version="1.0" encoding="utf-8"?>\n'
                   + ET.tostring(root, encoding="unicode"), encoding="utf-8")

    add_home_keyframe()

    print(f"✅ 已生成 {OUT}")
    print("   新增: 地板/工作桌/料箱/方块×2 | 相机×3 | home keyframe")


def add_home_keyframe() -> None:
    """按 joint/actuator 名字把参考工程的 home 姿态映射进我们的模型。

    为什么不手抄那串 48 个数字? —— qpos 顺序由树结构决定, 两边模型
    哪怕只差一个 body 顺序就全错。按名字映射是唯一可靠的办法。
    """
    mine = mujoco.MjModel.from_xml_path(str(OUT))
    expert = mujoco.MjModel.from_xml_path(str(EXPERT))
    kid = mujoco.mj_name2id(expert, mujoco.mjtObj.mjOBJ_KEY, "home")

    qpos = np.zeros(mine.nq)
    for j in range(expert.njnt):
        name = mujoco.mj_id2name(expert, mujoco.mjtObj.mjOBJ_JOINT, j)
        my_jid = mujoco.mj_name2id(mine, mujoco.mjtObj.mjOBJ_JOINT, name)
        if my_jid < 0:
            raise ValueError(f"参考工程有而我没有的关节: {name}")
        dim = 7 if expert.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        e_adr, m_adr = expert.jnt_qposadr[j], mine.jnt_qposadr[my_jid]
        qpos[m_adr:m_adr + dim] = expert.key_qpos[kid][e_adr:e_adr + dim]

    ctrl = np.zeros(mine.nu)
    for a in range(expert.nu):
        name = mujoco.mj_id2name(expert, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        my_aid = mujoco.mj_name2id(mine, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if my_aid >= 0:
            ctrl[my_aid] = expert.key_ctrl[kid][a]

    tree = ET.parse(OUT)
    key = ET.SubElement(tree.getroot(), "keyframe")
    ET.SubElement(key, "key", {
        "name": "home",
        "qpos": " ".join(f"{v:.6g}" for v in qpos),
        "ctrl": " ".join(f"{v:.6g}" for v in ctrl),
    })
    ET.indent(tree, space="  ")
    OUT.write_text('<?xml version="1.0" encoding="utf-8"?>\n'
                   + ET.tostring(tree.getroot(), encoding="unicode"),
                   encoding="utf-8")


def self_test() -> None:
    model = mujoco.MjModel.from_xml_path(str(OUT))
    data = mujoco.MjData(model)

    print(f"\n── 自测 ──")
    print(f"   nq={model.nq} (期望 48)   nu={model.nu}   nsensor={model.nsensor} (期望 142)")
    assert model.nq == 48 and model.nkey == 1 and model.nsensor == 142

    # 相机三路齐全?
    for cam in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        assert cid >= 0, f"相机缺失: {cam}"
    print("   相机: head_rgb + 双腕 wrist_rgb ✅")

    # 回到 home, 保持 3 秒: 方块落桌稳住, 机器人姿态不散
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.ctrl[:] = model.key_ctrl[home]
    for _ in range(1000):               # 1s: 方块下落/弹跳的暂态
        mujoco.mj_step(model, data)
    q_ref = data.qpos.copy()
    for _ in range(2000):               # 2s: 看稳态
        mujoco.mj_step(model, data)

    all_ok = True
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"red_cube_{side}")
        z = data.xpos[bid][2]
        settled = 0.64 < z < 0.70       # 桌面 0.641 + 半边 0.02 = 静置 0.661
        all_ok &= settled
        print(f"   {side:5s}方块: z={z:.3f} m {'✅ 落桌稳住' if settled else '❌ 没落到桌上'}")

    drift = float(np.abs(data.qpos[:34] - q_ref[:34]).max())
    hold = drift < 0.05
    all_ok &= hold
    print(f"   机器人 3s 姿态漂移: {drift:.4f} rad {'✅ 站得住' if hold else '❌ 姿态散了'}")
    print(f"   接触点: {data.ncon} (>0 表示方块压在桌上)")
    print("   结论:", "✅ 场景可用" if all_ok and data.ncon > 0 else "❌ 还有问题")


if __name__ == "__main__":
    build()
    self_test()
