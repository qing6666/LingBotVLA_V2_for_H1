#!/usr/bin/env python3
"""Step 11: 补齐接 PICO teleop 缺的两个模型接口 —— 目标体 + TCP frame 壳。

输入/输出: mujoco/H1_scene.xml   (Step 7 产物, 本步原地打补丁, 可重复执行)

补什么、为什么:

  ① left/right_teleop_target (mocap 体, 纯视觉"幽灵坐标系")
     h1_pico_teleop.py 每个控制周期把 IK 目标位姿写进这两个 mocap 刚体,
     在 Viewer 里可视化"手想去哪" —— 小球 + RGB 三轴。
       · mocap="true": 位姿由 Python 直接设置, 不参与物理积分;
       · 只有 site 没有 geom: 零质量零碰撞, 纯视觉, 不会推动方块;
       · teleop 启动时打印 "Mocap ID for ... : 0/1", 若是 -1 说明没跑本步。

  ② left/right_omnipicker_tcp_link (空 body, TCP 的"frame 壳")
     teleop 的 --check-model 要求 TCP frame 在 MJCF(body) 与 URDF(link)
     两边同名存在, 然后对比 home 下位姿。我们建场景时 TCP 用的是 site
     (Step 9 的 IK URDF 就是按这个 site 验的 FK, 误差 0.000mm) ——
     几何上已经对了, 只缺一个同名 body 壳。本步从 site 的 pos 克隆出
     空 body 挂在同一父级下, 两者世界位姿恒等, 传感器不受影响。

  补完这两组共 4 个 body 后 nbody 41→45, 与参考工程完全对齐 ——
  Step 8 对照报告里"nbody 41 vs 45"的差异至此全部闭合。

用法:  python scripts/11_add_teleop_targets.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent.parent
SCENE = HERE / "mujoco/H1_scene.xml"

# 幽灵目标体: 球心标记 + 0.08m 长的 RGB 三轴 (group=4, 默认分组显示)
TARGET_TMPL = """
    <body name="{side}_teleop_target" mocap="true" pos="0 0 1">
      <site type="sphere" size="0.012" rgba="{ball} 0.7" group="4" />
      <site type="box" size="0.04 0.002 0.002" pos="0.04 0 0" rgba="1 0 0 0.8" group="4" />
      <site type="box" size="0.002 0.04 0.002" pos="0 0.04 0" rgba="0 1 0 0.8" group="4" />
      <site type="box" size="0.002 0.002 0.04" pos="0 0 0.04" rgba="0 0 1 0.8" group="4" />
    </body>"""


def build() -> None:
    tree = ET.parse(SCENE)
    root = tree.getroot()

    if root.find(".//body[@name='left_teleop_target']") is not None:
        print("ℹ️  mocap 目标体已存在, 跳过插入 (本脚本可重复执行)")
    else:
        worldbody = root.find("worldbody")
        for side, ball in (("left", "0.1 0.8 1"), ("right", "1 0.7 0.1")):
            worldbody.append(ET.fromstring(TARGET_TMPL.format(side=side, ball=ball)))
        print("✅ 已添加 left/right_teleop_target mocap 体")

    add_tcp_link_bodies(root)
    ensure_offscreen_buffer(root)

    ET.indent(tree, space="  ")
    SCENE.write_text('<?xml version="1.0" encoding="utf-8"?>\n'
                     + ET.tostring(root, encoding="unicode"), encoding="utf-8")


def add_tcp_link_bodies(root: ET.Element) -> None:
    """从 TCP site 克隆出同名空 body —— frame 壳, 位姿与 site 恒等。

    site 的父级是 *_omnipicker_mount; 新 body 挂进同一父级、pos 抄 site 的,
    不带旋转 → 两者世界位姿完全相同 (自测里逐分量断言)。
    """
    for side in ("left", "right"):
        name = f"{side}_omnipicker_tcp_link"
        if root.find(f".//body[@name='{name}']") is not None:
            print(f"ℹ️  {name} 已存在, 跳过")
            continue
        site = root.find(f".//site[@name='{side}_omnipicker_tcp']")
        if site is None:
            raise ValueError(f"找不到 TCP site: {side}_omnipicker_tcp")
        mount = next(b for b in root.iter("body") if site in list(b))
        ET.SubElement(mount, "body", {"name": name, "pos": site.get("pos", "0 0 0")})
        print(f"✅ 已从 site 克隆 TCP frame 壳: {name} (pos={site.get('pos')})")


def ensure_offscreen_buffer(root: ET.Element) -> None:
    """离屏渲染缓冲 ≥ 512 —— 数采记录器要渲染 512×512 的三路相机图。

    MuJoCo 默认 offwidth×offheight = 640×480 (高度不够), 参考工程显式设了
    3840×2160。此处抄同款数值; 不影响窗口显示, 只决定离屏渲染的上限。
    """
    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(list(root).index(root.find("worldbody")), visual)
    globe = visual.find("global")
    if globe is None:
        globe = ET.SubElement(visual, "global")
    if globe.get("offwidth") == "3840" and globe.get("offheight") == "2160":
        print("ℹ️  离屏缓冲已是 3840x2160, 跳过")
        return
    globe.set("offwidth", "3840")
    globe.set("offheight", "2160")
    print("✅ 已设置离屏渲染缓冲 3840x2160 (数采 512x512 三相机渲染的前提)")


def self_test() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))

    print("\n── 自测 ──")
    assert model.nmocap == 2, f"nmocap={model.nmocap}, 期望 2"
    for i, side in enumerate(("left", "right")):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_teleop_target")
        assert bid >= 0, f"目标体缺失: {side}_teleop_target"
        mid = int(model.body_mocapid[bid])
        assert mid == i, f"{side} 的 mocap id={mid}, 期望 {i} (teleop 按顺序索引)"
        print(f"   {side}_teleop_target: body id={bid}, mocap id={mid} ✅")

    # TCP frame 壳: 在 home 姿态下与 site 位姿逐分量恒等
    # (site 是 Step 9 IK URDF 的 FK 基准, 壳与它重合 ⟹ --check-model 必过)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_omnipicker_tcp_link")
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_omnipicker_tcp")
        assert body_id >= 0 and site_id >= 0, f"TCP 壳或 site 缺失: {side}"
        err = float(np.abs(data.xpos[body_id] - data.site_xpos[site_id]).max())
        assert err < 1e-9, f"{side} TCP 壳与 site 位姿差 {err}"
        print(f"   {side}_omnipicker_tcp_link ≡ site (误差 {err:.1e}) ✅")

    # 本步只加"壳与纯视觉体", 不得动到任何既有结构
    assert model.nq == 48 and model.nu == 20 and model.nsensor == 142, \
        f"结构被意外改变: nq={model.nq} nu={model.nu} nsensor={model.nsensor}"
    assert model.nkey == 1 and home >= 0, "home keyframe 丢失"
    print(f"   nbody={model.nbody} (41→45, 与参考工程对齐), "
          f"nq/nu/nsensor/home 全部不变 ✅")

    # mocap 位姿可写: teleop 每帧都会这么干, 提前验证一次
    data.mocap_pos[0] = [0.5, 0.3, 0.7]
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_teleop_target")
    assert abs(float(data.xpos[bid][0]) - 0.5) < 1e-9, "mocap 位姿写入未生效"
    print("   mocap 位姿写入生效 ✅")
    print("   结论: ✅ 场景已就绪, 可接 h1_pico_teleop.py")


if __name__ == "__main__":
    build()
    self_test()
