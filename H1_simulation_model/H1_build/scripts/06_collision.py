#!/usr/bin/env python3
"""Step 6: 碰撞体改造 —— 视觉/碰撞分离, 解决"手指自己卡自己"。

输入: mujoco/H1_grippers.xml    (Step 5 产物: 手指 STL 网格互相穿插碰撞)
输出: mujoco/H1_collision.xml

背景: 四连杆夹爪的耦合杆(loop)和手指天然交叉, 需要视觉/碰撞分离 +
"机器人永不自碰"的碰撞规则, 否则手指互相卡死。
标准解法(也是参考工程的做法):
  ① 所有 mesh 几何体 → class "visual": contype=0 conaffinity=0, 只渲染不碰撞
  ② 碰撞体分三种:
       手臂/躯干   → AABB 包围盒(满包围盒的 48%), class "collision"
       手指+耦合杆 → 直接拿 STL 凸包当碰撞体 type="mesh", class "finger_collision"
       夹爪座(mount, mesh=omnipicker_base_link) → 同手臂: 48% AABB 盒
         (参考工程 *_mount_collision* 即此盒; 名字也逐字一致)
  ③ 碰撞规则设计(自卡问题的真正解在这里, 跟碰撞体形状无关):
       visual:          c=0 a=0  和谁都不碰(纯外观)
       collision(本体): c=0 a=1  只和环境(contype=1)碰, 机器人部位之间永不自碰
       finger_collision: 同上但高摩擦, 抓东西用
     —— 自碰撞交给关节限位和 IK 管, 物理引擎专心算"机器人 vs 环境"。

2026-08-19 修复"抓取穿模+弹飞": 手指碰撞体从 AABB 盒改为 mesh。
AABB 盒比真实手指胖一圈(loop 件 y 向有 43mm 宽, 正好横在抓取口里),
配硬接触参数后盒角先蹭到方块、再把它弹飞。参考工程的手指全部用
type="mesh" 碰撞体(实测 16 个 = 8 种网格 x 左右手, 含两个 loop 件),
只有 omnipicker_base_link 走的是手臂同款 48% AABB 盒(挂在 mount 体上,
名字 *_mount_collision*) —— 本步与之逐项对齐 (21 盒 + 16 mesh)。

AABB 表来源: 参考工程用 MuJoCo 编译后实测的各 mesh 包围盒 (center, 全尺寸)。
自测: 夹爪开合测试必须双侧达标(Step 5 左侧失败的根因就是手指自碰)。
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "mujoco/H1_grippers.xml"
OUT = HERE / "mujoco/H1_collision.xml"

# mesh AABB: 名字 -> ((中心x,y,z), (全尺寸x,y,z)) 米。参考工程实测值。
MESH_BOXES = {
    "base_link": ((0.00063, 0.00012, -0.02628), (0.15423, 0.17861, 0.69047)),
    "waist_Link_j1": ((0, -0.00813, -0.00543), (0.12494, 0.17019, 0.23899)),
    "waist_link_J2": ((0, -0.00042, -0.02175), (0.12504, 0.29096, 0.46958)),
    "left_link_J1": ((0.00160, 0, 0.00878), (0.08460, 0.09501, 0.12933)),
    "Left_Link_J2": ((0, 0.00489, 0.00444), (0.09501, 0.11429, 0.15099)),
    "Left_Link_J3": ((0.00143, -0.00268, 0.00176), (0.08795, 0.09017, 0.25136)),
    "left_Link_j4": ((0, -0.00922, 0.00104), (0.08199, 0.12249, 0.15443)),
    "Left_Link_J5": ((0, -0.00099, 0.00103), (0.06999, 0.07544, 0.23608)),
    "Left_Link_J6": ((0.00120, -0.00112, 0.00218), (0.07726, 0.09339, 0.12764)),
    "Left_Link_J7": ((0, 0.00387, 0.00019), (0.06204, 0.07602, 0.09300)),
    "Right_Link_J1": ((-0.00160, 0, -0.00878), (0.08460, 0.09501, 0.12933)),
    "Right_Link_J2": ((0.00009, 0.00551, 0.00046), (0.09524, 0.11477, 0.15141)),
    "Right_Link_J3": ((0.00143, -0.00268, 0.00176), (0.08795, 0.09017, 0.25136)),
    "Right_Link_J4": ((0, -0.00922, 0.00104), (0.08199, 0.12249, 0.15443)),
    "Right_Link_J5": ((0, 0.00099, 0.00103), (0.06999, 0.07544, 0.23608)),
    "Right_Link_J6": ((-0.00120, 0.00112, 0.00218), (0.07726, 0.09339, 0.12764)),
    "Right_Link_J7": ((0, 0.00387, -0.00019), (0.06204, 0.07602, 0.09300)),
    "head_j1": ((0, 0.00228, -0.00574), (0.06202, 0.08253, 0.13351)),
    "Head_Link_j2": ((0.00009, 0.00114, 0.00680), (0.06096, 0.10985, 0.12375)),
    "omnipicker_base_link": ((-0.00010, 0, 0.00707), (0.05721, 0.07936, 0.08371)),
    "omnipicker_narrow1_Link": ((-0.00080, -0.00005, 0.00112), (0.02564, 0.03685, 0.05476)),
    "omnipicker_narrow2_Link": ((-0.00173, 0, -0.00116), (0.01399, 0.01106, 0.03598)),
    "omnipicker_narrow3_Link": ((0.00079, 0.00011, 0.00664), (0.02229, 0.02606, 0.05761)),
    "omnipicker_narrow_loop_Link": ((0, -0.00077, 0.00017), (0.01357, 0.04317, 0.05104)),
    "omnipicker_wide1_Link": ((0.00091, -0.00005, 0.00110), (0.02559, 0.03686, 0.05483)),
    "omnipicker_wide2_Link": ((0.00173, 0, -0.00116), (0.01399, 0.01106, 0.03598)),
    "omnipicker_wide3_Link": ((-0.00038, -0.00324, 0), (0.02223, 0.05767, 0.06038)),
    "omnipicker_wide_loop_Link": ((-0.00010, 0.00015, 0.00019), (0.01343, 0.04310, 0.05105)),
}

fmt = lambda v: " ".join(f"{x:.6g}" for x in v)


def build() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()

    # ① defaults 里加三个 geom 档位
    default = root.find("default")
    for cls, attrs in {
        "visual": 'contype="0" conaffinity="0" group="2"',
        "collision": ('contype="0" conaffinity="1" group="3" rgba="0.2 0.8 0.2 0.15" '
                      'friction="0.8 0.02 0.001" condim="3" '
                      'solref="0.01 1" solimp="0.9 0.95 0.001"'),
        # 指尖参数与参考工程逐字一致 —— margin(提前 1mm 触发) + 高摩擦 +
        # condim=6 + 硬接触是"抓得住、不穿模"的四件套; 2026-08-18 实机
        # 遥操作抓取穿模后对齐 (当时自建的软参数 solref=0.005/solimp=0.95)。
        "finger_collision": ('contype="0" conaffinity="1" group="3" rgba="1 0.25 0.1 0.2" '
                             'priority="2" margin="0.001" friction="2.0 0.05 0.005" condim="6" '
                             'solref="0.001 1" solimp="0.99 0.9999 0.0001"'),
    }.items():
        sub = ET.SubElement(default, "default", {"class": cls})
        sub.append(ET.fromstring(f'<geom {attrs}/>'))

    # ② 逐个 mesh geom: 改成 visual 档 + 按部位补碰撞体
    n_visual = n_box = n_mesh = 0
    for container in [root.find("worldbody"), *root.findall(".//body")]:
        owner = container.get("name", "world")
        for geom in list(container.findall("geom")):
            mesh = geom.get("mesh")
            if not mesh:
                continue
            geom.set("class", "visual")
            for stale in ("contype", "conaffinity", "collision"):   # 清掉旧碰撞属性
                geom.attrib.pop(stale, None)
            n_visual += 1

            if mesh not in MESH_BOXES:
                raise ValueError(f"AABB 表里没有这个 mesh: {mesh}")
            center, full = MESH_BOXES[mesh]
            finger = (mesh.startswith("omnipicker_")
                      and not mesh.endswith("base_link"))
            if finger:
                # 手指 + 耦合杆: STL 凸包碰撞体, 与真实指形一致 (不穿模的关键)
                container.append(ET.Element("geom", {
                    "name": owner + "_collision", "type": "mesh",
                    "mesh": mesh, "class": "finger_collision",
                }))
                n_mesh += 1
                # 指尖触觉 site 只给 6 个真指尖件(narrow1/2/3 + wide1/2/3) x2 手;
                # loop 耦合杆不是指尖, 不埋点 —— Step 7 按 touch_site 数量建
                # 触觉传感器, 多埋会把 nsensor 顶过 142 的对齐基准。
                if "loop" not in mesh:
                    ET.SubElement(container, "site", {
                        "name": owner + "_touch_site", "type": "box",
                        "pos": fmt(center),
                        "size": fmt(tuple(0.51 * v for v in full)),
                        "rgba": "1 0.2 0.1 0.08", "group": "4",
                    })
            else:
                box = ET.Element("geom", {
                    "name": owner + "_collision", "type": "box",
                    "class": "collision",
                    "pos": fmt(center),
                    "size": fmt(tuple(0.48 * v for v in full)),   # 半尺寸 = 48%全尺寸
                })
                container.append(box)
                n_box += 1

    ET.indent(tree, space="  ")
    OUT.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    print(f"✅ 已生成 {OUT}")
    print(f"   visual mesh: {n_visual} 个 | 手臂碰撞盒: {n_box} 个 | 手指 mesh 碰撞体: {n_mesh} 个 (期望 16)")


def self_test() -> None:
    """重跑 Step 5 失败的夹爪开合测试 —— 现在应该双侧都过。"""
    model = mujoco.MjModel.from_xml_path(str(OUT))
    data = mujoco.MjData(model)
    print(f"\n── 自测: 夹爪开合(Step 5 的左爪失败项) ──")
    all_ok = True
    for side in ("left", "right"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"{side}_omnipicker_gripper_opening")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{side}_omnipicker_hand_narrow1_joint")
        d = mujoco.MjData(model)
        d.ctrl[aid] = 1.0
        for _ in range(2000):                    # 2 秒: 限幅力下的蠕变需要时间
            mujoco.mj_step(model, d)             # (500 步只到一半, 别再误判卡死)
        q_open = d.qpos[model.jnt_qposadr[jid]]
        d.ctrl[aid] = 0.0
        for _ in range(2000):
            mujoco.mj_step(model, d)
        q_close = d.qpos[model.jnt_qposadr[jid]]
        moved = abs(q_open - q_close)
        ok = moved > 1.0 and abs(q_open - (-1.2)) < 0.15  # ctrl=1 → narrow1 ≈ -1.2
        all_ok &= ok
        print(f"   {side:5s}爪: 张开{q_open:+.3f} → 闭合{q_close:+.3f} "
              f"行程{moved:.3f} rad {'✅' if ok else '❌'}")
        # 自碰检查: 静止状态下不应有任何接触(还没有环境物体)
        mujoco.mj_forward(model, d)
        print(f"        静止接触点: {d.ncon} (期望 0, 机器人不自碰)")
    print("   结论:", "✅ 全部通过" if all_ok else "❌ 还有问题")


if __name__ == "__main__":
    build()
    self_test()
