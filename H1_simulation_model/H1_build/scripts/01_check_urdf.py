#!/usr/bin/env python3
"""Step 1: URDF 体检 —— 拿到 URDF 后第一件事，全面检查潜在问题。

用法: python scripts/01_check_urdf.py [urdf路径]
检查项:
  1. XML 是否合法
  2. link/joint 数量与类型
  3. 每个关节的 limit（限位/力矩/速度是否为 0 —— SolidWorks 导出常见坑）
  4. mesh 路径（package:// 是 ROS 协议，MuJoCo 不认）
  5. 阻尼 dynamics 是否缺失
  6. 每个连杆是否有惯量（仿真必需）
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# 关节分类: 用于给不同部位填不同电机参数（对应电机 BOM）
def classify(name: str) -> str:
    if name.startswith("waist") or name == "J1":
        return "腰部"
    if name.startswith(("Left", "Right")):
        return "手臂"
    return "头部"


def main() -> None:
    # 默认路径以脚本位置为基准 (H1_build/), 在任何目录下运行都能找到
    here = Path(__file__).resolve().parent.parent
    urdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "urdf/H1_raw.urdf"
    print(f"=== URDF 体检: {urdf_path} ===\n")

    # ── 检查 1: XML 合法性 ──────────────────────────────
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as e:
        sys.exit(f"❌ XML 解析失败: {e}")
    print(f"✅ XML 合法 | robot name = {root.get('name')}")

    # ── 检查 2: 数量统计 ────────────────────────────────
    links = root.findall("link")
    joints = root.findall("joint")
    movable = [j for j in joints if j.get("type") != "fixed"]
    print(f"✅ 结构 | {len(links)} links, {len(joints)} joints "
          f"(可动 {len(movable)}: " +
          ", ".join(f"{j.get('name')}({j.get('type')})" for j in movable) + ")")

    # ── 检查 3/5: 关节限位与阻尼 ────────────────────────
    print("\n── 关节参数 ──")
    problems = 0
    for j in movable:
        name = j.get("name")
        lim = j.find("limit")
        dyn = j.find("dynamics")
        parts = [classify(name)]
        if lim is None:
            parts.append("❌无limit标签"); problems += 1
        else:
            # 注意: lower=0 可能是合法限位(如肘部单侧关节)，
            # 只有"上下限都为0"(锁死)或 effort/velocity 为 0 才是真问题
            frozen = (float(lim.get("lower", 0)) == 0.0 and
                      float(lim.get("upper", 0)) == 0.0)
            weak = [k for k in ("effort", "velocity")
                    if float(lim.get(k, 0)) == 0.0]
            if frozen or weak:
                parts.append(f"❌{'锁死' if frozen else ''}{'+'.join(weak)}=0"); problems += 1
            else:
                parts.append(f"limit[{lim.get('lower')},{lim.get('upper')}] "
                             f"effort={lim.get('effort')}")
        if dyn is None or float(dyn.get("damping", 0)) == 0:
            parts.append("⚠️无阻尼"); problems += 1
        print(f"  {name:12s} {' | '.join(parts)}")

    # ── 检查 4: mesh 路径 ───────────────────────────────
    print("\n── mesh 路径 ──")
    meshes = [m.get("filename") for m in root.findall(".//mesh")]
    bad = [m for m in meshes if m.startswith("package://")]
    if bad:
        problems += 1
        print(f"  ❌ {len(bad)}/{len(meshes)} 使用 package:// 协议 (MuJoCo 不认)")
        print(f"     例: {bad[0]}")
    else:
        print(f"  ✅ {len(meshes)} 个 mesh 路径正常")

    # ── 检查 6: 惯量 ────────────────────────────────────
    no_inertial = [l.get("name") for l in links
                   if l.find("inertial") is None or
                   float(l.find("inertial/mass").get("value", 0)) <= 0]
    if no_inertial:
        problems += 1
        print(f"\n❌ 以下 link 缺惯量/质量: {no_inertial}")
    else:
        print(f"\n✅ 全部 {len(links)} 个 link 均有惯量")

    # ── 总结 ───────────────────────────────────────────
    print(f"\n=== 结论: 发现 {problems} 类问题，需要修复后才能仿真 ===")


if __name__ == "__main__":
    main()
