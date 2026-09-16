#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
13_edit_home_keyframe.py —— 修改 H1_scene.xml 里 home keyframe 的关节初始位置

home keyframe 是整个工程的"姿态唯一来源"：
  * 遥操作 h1_pico_teleop.py 的初始姿态、B 键回零、头/腰中性姿态都从这里读；
  * 推理 h1_smolvla_inference_rollout.py 每回合 reset 也回到这里。
所以想改机器人开局姿态（比如让头部相机低头俯视桌面），只改这里就够了，
不用动任何 Python 代码。

本脚本解决"48 个数字的长串没法手改"的问题：
  * 按【关节名 → 序号】映射（加载模型后用 mujoco 查地址，绝不手数）；
  * 一个关节的初始角度会同时写入两处——qpos（初始关节角）和对应
    <关节名>_position 执行器的 ctrl（回 home 后执行器拉住这个角度不漂移）；
  * 只重写 <key name="home"> 那一行，文件其余部分一个字节都不动；
  * 首次修改前自动备份整个 XML（H1_scene.xml.bak，--restore 可还原）。

用法（在 lerobot312 环境，任何目录都能跑）：
    # 1) 查看当前 home keyframe 里每个关节的初始值（改前先看这个）
    python H1_simulation_model/H1_build/scripts/13_edit_home_keyframe.py --show

    # 2) 改配置区里的 CHANGES 字典后直接运行，写入
    python H1_simulation_model/H1_build/scripts/13_edit_home_keyframe.py

    # 3) 临时改一处（不用编辑文件），可重复多个 --set
    python .../13_edit_home_keyframe.py --set Head_j2=0.4 --set waist_J1=0.1

    # 4) 出问题想回到最初备份
    python .../13_edit_home_keyframe.py --restore

注意：
  * 关节名大小写敏感（Right_j6 的 j 是小写，历史命名，勿"修正"）；
  * 值单位是弧度，会自动限幅到关节/执行器的 range，越界会打印警告；
  * 两个红方块是 freejoint（7 维位姿），不在本脚本范围；
  * 改完即生效（模型加载时读 XML），无需重新生成场景。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import mujoco

# ==================== 路径与常量 ====================

H1_ROOT = Path(__file__).resolve().parents[1]          # .../H1_build
XML_PATH = H1_ROOT / "mujoco" / "H1_scene.xml"
BACKUP_PATH = XML_PATH.with_suffix(".xml.bak")          # H1_scene.xml.bak
KEYFRAME_NAME = "home"

# ==================== 配置区：全关节默认值总表 ====================
#
# 键 = MJCF 关节名；值 = 初始角度（弧度）。下表已把 18 个上身关节全部列出
# 并填上当前默认值——整表原样运行等于无操作，改哪行就动哪个关节。
# 手指关节不在此表：它们没有独立执行器，由下方 GRIPPER_OPENINGS 的
# 开合 ctrl 经 equality 统一驱动。
# 常用方向参考：
#   Head_j2   头部俯仰（带动 head_rgb 相机）：正=低头俯视，负=抬头，范围 ±0.7854
#   head_j1   头部左右转，范围 ±3.1416
#   waist_J1  腰部回转 / waist_J2 腰部俯仰（0 ~ 1.5708）
#   Left_J4 / Right_J4  左右肘关节，默认弯曲 90°（1.5708）
CHANGES: dict[str, float] = {
    # ── 腰 ──
    "waist_J1": 0.0,       # 腰部回转
    "waist_J2": 0.0,       # 腰部俯仰
    # ── 左臂（肩 → 肘 → 腕）──
    "Left_J1": 0.0,
    "Left_J2": 0.0,
    "Left_J3": 0.0,
    "Left_J4": 1.5708,     # 左肘，默认弯曲 90°
    "Left_J5": 0.0,
    "Left_J6": 0.0,
    "Left_J7": 0.0,
    # ── 右臂（Right_j6 的小写 j 是模型历史命名，勿"修正"）──
    "Right_J1": 0.0,
    "Right_J2": 0.0,
    "Right_J3": 0.0,
    "Right_J4": 1.5708,    # 右肘，默认弯曲 90°
    "Right_J5": 0.0,
    "Right_j6": 0.0,       # 注意小写 j
    "Right_J7": 0.0,
    # ── 头（衔接 head_rgb 相机）──
    "head_j1": 0.0,        # 头部左右转
    "Head_j2": 0.0,        # 头部俯仰：0 → 0.4（2026-08-20 改，开局俯视桌面）
}

# 夹爪初始开合（0=闭合，1=全开）。夹爪没有 _position 执行器（是归一化开合 ctrl），
# 所以单独在这里配。
GRIPPER_OPENINGS: dict[str, float] = {
    "left": 0.0,
    "right": 0.0,
}


# ==================== 工具函数 ====================


def find_home_key_line(text: str) -> tuple[int, str]:
    """在 XML 文本里定位 <key name="home"> 所在行，返回 (行号, 行内容)。"""
    for i, line in enumerate(text.splitlines(keepends=True)):
        if f'<key name="{KEYFRAME_NAME}"' in line:
            return i, line
    raise SystemExit(f"[error] {XML_PATH.name} 里找不到 <key name=\"{KEYFRAME_NAME}\">")


def patch_attribute(line: str, attr: str, token_index: int, value: str) -> str:
    """把行内 attr="..." 属性串的第 token_index 个数字替换为 value，行其余部分不动。"""
    mat = re.search(attr + r'="([^"]+)"', line)
    if mat is None:
        raise SystemExit(f"[error] keyframe 行里找不到属性 {attr}")
    tokens = mat.group(1).split()
    if len(tokens) <= token_index:
        raise SystemExit(f"[error] {attr} 只有 {len(tokens)} 个数，索引 {token_index} 越界")
    tokens[token_index] = value
    return line[: mat.start(1)] + " ".join(tokens) + line[mat.end(1) :]


def clamp_to_range(value: float, lo: float, hi: float, what: str) -> float:
    """把值限幅到 [lo, hi]，越界时打印警告（不报错，按限幅值写入）。"""
    if value < lo or value > hi:
        clamped = min(max(value, lo), hi)
        print(f"[warn] {what}: {value} 超出范围 [{lo}, {hi}]，按 {clamped:.4f} 写入")
        return clamped
    return value


def joint_position_actuator(model: mujoco.MjModel, joint_name: str) -> int:
    """返回关节对应的 <joint>_position 执行器 id；没有则返回 -1。"""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_position")


def show_current(model: mujoco.MjModel) -> None:
    """打印 home keyframe 里全部关节的初始 qpos / ctrl，方便挑要改的关节名。"""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    print(f"当前 home keyframe 各关节初始值（{XML_PATH.name}）：\n")
    print(f"{'关节名':<30} {'qpos(rad)':>10} {'ctrl(rad)':>10}   关节范围")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue  # 跳过 freejoint（方块）
        qadr = int(model.jnt_qposadr[j])
        aid = joint_position_actuator(model, name)
        ctrl_val = model.key_ctrl[kid, aid] if aid >= 0 else float("nan")
        lo, hi = model.jnt_range[j]
        mark = "  ←" if aid >= 0 and abs(model.key_qpos[kid, qadr] - ctrl_val) > 1e-9 else ""
        print(f"{name:<30} {model.key_qpos[kid, qadr]:>10.4f} {ctrl_val:>10.4f}   [{lo:+.4f}, {hi:+.4f}]{mark}")
    print("\n夹爪开合 ctrl（0=闭合 1=全开）：")
    for side in ("left", "right"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_omnipicker_gripper_opening")
        print(f"  {side:<6} {model.key_ctrl[kid, aid]:.4f}")
    if any(True for j in range(model.njnt)
           if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
           and (aid := joint_position_actuator(model, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))) >= 0
           and abs(model.key_qpos[kid, model.jnt_qposadr[j]] - model.key_ctrl[kid, aid]) > 1e-9):
        print("\n（标 ← 的行 qpos 与 ctrl 不一致：回 home 后执行器会把它拉到 ctrl 值）")


def apply_changes(changes: dict[str, float], grippers: dict[str, float]) -> None:
    """把 CHANGES / GRIPPER_OPENINGS 写入 home keyframe 行，并回读验证。"""
    if not changes and not grippers:
        print("[info] 配置区为空，什么都没改。用 --show 查看当前值，或在 CHANGES 里加关节。")
        return

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)

    # 1) 按名字解析每项要写的 (qpos索引, ctrl索引, 限幅后的值)
    writes: list[tuple[str, int, int, float]] = []  # (名字, qpos_idx, ctrl_idx, 值)
    for joint_name, value in changes.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise SystemExit(f"[error] 模型里没有关节 '{joint_name}'（注意大小写，如 Right_j6）")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise SystemExit(f"[error] '{joint_name}' 不是单自由度 hinge 关节，不支持本脚本")
        qpos_idx = int(model.jnt_qposadr[jid])
        qlo, qhi = model.jnt_range[jid]
        value = clamp_to_range(float(value), qlo, qhi, f"{joint_name} qpos")
        ctrl_idx = joint_position_actuator(model, joint_name)
        if ctrl_idx >= 0:
            clo, chi = model.actuator_ctrlrange[ctrl_idx]
            value = clamp_to_range(value, clo, chi, f"{joint_name} ctrl")
        else:
            print(f"[warn] '{joint_name}' 没有 {joint_name}_position 执行器，只改 qpos 不改 ctrl")
            ctrl_idx = -1
        writes.append((joint_name, qpos_idx, ctrl_idx, value))

    for side, opening in grippers.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_omnipicker_gripper_opening")
        if aid < 0:
            raise SystemExit(f"[error] 没有夹爪执行器 '{side}_omnipicker_gripper_opening'")
        value = clamp_to_range(float(opening), 0.0, 1.0, f"{side} 夹爪开合")
        writes.append((f"{side} 夹爪开合", -1, aid, value))

    # 2) 首次修改前备份（保留最初版本；--restore 用它还原）
    if not BACKUP_PATH.exists():
        shutil.copy2(XML_PATH, BACKUP_PATH)
        print(f"[backup] 已备份原始文件 -> {BACKUP_PATH.name}")
    else:
        print(f"[backup] 备份已存在（保留最初版本）: {BACKUP_PATH.name}")

    # 3) 只重写 home keyframe 那一行
    text = XML_PATH.read_text()
    line_no, line = find_home_key_line(text)
    lines = text.splitlines(keepends=True)
    for name, qpos_idx, ctrl_idx, value in writes:
        print(f"[edit] {name}: -> {value:.4f}")
        if qpos_idx >= 0:
            line = patch_attribute(line, "qpos", qpos_idx, f"{value:.6g}")
        if ctrl_idx >= 0:
            line = patch_attribute(line, "ctrl", ctrl_idx, f"{value:.6g}")
    lines[line_no] = line
    XML_PATH.write_text("".join(lines))

    # 4) 重新加载模型回读验证
    model2 = mujoco.MjModel.from_xml_path(str(XML_PATH))
    kid2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    print("[verify] 回读验证：")
    ok = True
    for name, qpos_idx, ctrl_idx, value in writes:
        got_q = model2.key_qpos[kid2, qpos_idx] if qpos_idx >= 0 else None
        got_c = model2.key_ctrl[kid2, ctrl_idx] if ctrl_idx >= 0 else None
        good = (got_q is None or abs(got_q - value) < 1e-6) and (got_c is None or abs(got_c - value) < 1e-6)
        ok = ok and good
        print(f"  {name}: qpos={got_q} ctrl={got_c} {'✓' if good else '✗ 不一致!'}")
    if not ok:
        raise SystemExit("[error] 回读验证失败，请用 --restore 还原后重试")
    print(f"[done] 修改完成并验证通过：{XML_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="修改 home keyframe 的关节初始位置（按名字映射，不手数数字）")
    parser.add_argument("--show", action="store_true", help="只查看当前各关节初始值，不修改")
    parser.add_argument("--set", action="append", default=[], metavar="关节=弧度",
                        help="临时修改一处（可重复），例：--set Head_j2=0.4")
    parser.add_argument("--restore", action="store_true", help="从 .bak 备份还原 XML")
    args = parser.parse_args()

    if args.restore:
        if not BACKUP_PATH.exists():
            raise SystemExit(f"[error] 没有备份文件 {BACKUP_PATH}")
        shutil.copy2(BACKUP_PATH, XML_PATH)
        print(f"[done] 已从 {BACKUP_PATH.name} 还原 {XML_PATH.name}")
        return

    if args.show:
        show_current(mujoco.MjModel.from_xml_path(str(XML_PATH)))
        return

    # --set 的临时改动叠加在配置区 CHANGES 之上（同名以 --set 为准）
    changes = dict(CHANGES)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"[error] --set 格式应为 关节=值，收到: {item}")
        name, _, value = item.partition("=")
        changes[name.strip()] = float(value)

    apply_changes(changes, dict(GRIPPER_OPENINGS))


if __name__ == "__main__":
    main()
