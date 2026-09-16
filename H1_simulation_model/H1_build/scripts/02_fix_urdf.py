#!/usr/bin/env python3
"""Step 2: 修复 URDF —— 把体检发现的三类问题全部修掉。

输入: urdf/H1_raw.urdf   (SolidWorks 原始导出)
输出: urdf/H1_fixed.urdf (可被 MuJoCo 编译的修复版)

修复内容:
  1. 关节限位/力矩/速度: 全 0 → 真实电机规格（提取自前辈修好的版本，
     对应电机 BOM: 腰部 HBN90-052/HBN110-087, 肩肘 HBN80-031/HBN70-010,
     腕/头 HBN52-005; effort 数值即型号里的额定力矩）
  2. 关节阻尼: 缺失 → 按电机型号分级（越大电机阻尼越大）
  3. mesh 路径: package://H1/meshes/ → ../meshes/ （MuJoCo 相对路径规则:
     相对【URDF 文件所在目录】解析, 我们的 urdf/ 与 meshes/ 同级）
  4. 关节改名: J1 → waist_J1 （原名含义不明, 与参考工程对齐）
"""

import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # H1_build/
RAW = HERE / "urdf/H1_raw.urdf"
OUT = HERE / "urdf/H1_fixed.urdf"

# 真实电机规格表: 关节名 -> (lower, upper, effort, velocity, damping)
# 来源: 参考工程 h1_for_import.urdf 的限位 + 电机 BOM 分级阻尼
SPEC = {
    # ── 腰部 (HBN90-052: 52Nm / HBN110-087: 87Nm) ──
    "waist_J1": (-3.1415, 3.1415, 52, 2.0944, 1.0),
    "waist_J2": (0.0, 1.5708, 87, 2.0944, 1.2),
    # ── 左臂 (J1/J2: HBN80-031=31Nm, J3/J4: HBN70-010=10Nm, J5-J7: HBN52-005=5Nm) ──
    "Left_J1":  (-3.1415, 3.1415, 31, 3.1416, 0.6),
    "Left_J2":  (-0.3490, 2.0944, 31, 3.1416, 0.6),
    "Left_J3":  (-1.5708, 1.5708, 10, 3.1416, 0.3),
    "Left_J4":  (0.0, 2.0944, 10, 3.1416, 0.3),
    "Left_J5":  (-1.5708, 1.5708, 5, 3.1416, 0.15),
    "Left_J6":  (-0.6982, 0.6982, 5, 3.1416, 0.15),
    "Left_J7":  (-1.3090, 0.6982, 5, 3.1416, 0.15),
    # ── 右臂 ──
    "Right_J1": (-3.1416, 3.1416, 31, 3.1416, 0.6),
    "Right_J2": (-0.3490, 2.0944, 31, 3.1416, 0.6),
    "Right_J3": (-1.5708, 1.5708, 10, 3.1416, 0.3),
    "Right_J4": (0.0, 2.0944, 10, 3.1416, 0.3),
    "Right_J5": (-1.5708, 1.5708, 5, 3.1416, 0.15),
    "Right_j6": (-0.6982, 0.6982, 5, 3.1416, 0.15),
    "Right_J7": (-0.6982, 1.3090, 5, 3.1416, 0.15),
    # ── 头部 (HBN52-005) ──
    "head_j1":  (-3.1416, 3.1416, 5, 3.1416, 0.15),
    "Head_j2":  (-0.7854, 0.7854, 5, 3.1416, 0.15),
}


def main() -> None:
    tree = ET.parse(RAW)
    root = tree.getroot()

    fixed_joints = fixed_meshes = 0

    for joint in root.findall("joint"):
        name = joint.get("name")
        if name == "J1":                       # 改成更清晰的名字
            joint.set("name", "waist_J1")
            name = "waist_J1"

        if name not in SPEC:
            continue
        lo, hi, effort, vel, damping = SPEC[name]

        # ① 补限位
        lim = joint.find("limit")
        lim.set("lower", f"{lo:g}")
        lim.set("upper", f"{hi:g}")
        lim.set("effort", f"{effort:g}")
        lim.set("velocity", f"{vel:g}")

        # ② 补阻尼（没有 dynamics 标签就创建）
        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping", f"{damping:g}")
        dyn.set("friction", "0.1")
        fixed_joints += 1

    # ③ 修 mesh 路径
    for mesh in root.findall(".//mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            mesh.set("filename", "../meshes/" + fn.rsplit("/", 1)[-1])
            fixed_meshes += 1

    ET.indent(tree, space="  ")
    OUT.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    print(f"✅ 修复完成 -> {OUT}")
    print(f"   关节: {fixed_joints} 个 (限位+力矩+速度+阻尼)")
    print(f"   mesh 路径: {fixed_meshes} 处 (package:// -> ../meshes/)")
    print(f"   改名: J1 -> waist_J1")


if __name__ == "__main__":
    main()
