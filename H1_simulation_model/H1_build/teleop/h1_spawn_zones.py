# -*- coding: utf-8 -*-
"""v4 方块出生位置随机带 —— 数采 A 键重置与闭环 rollout 共用同一份(单一事实来源)。

背景
----
v3 的 148 条数据全部出生在 home keyframe 固定点 (0.65, ±0.30),模型只见过一种逼近
几何;且终末动作风格不一致(进近方向/快慢各异),flow-matching 的条件均值被多风格
摊薄,实测闭环"每段终末动作都是缩小版"(见 rollout --action-gain 诊断链)。
v4 用"位置随机 + 风格统一(正上方进近、终末慢速)"同时解决这两点。

带的几何依据(H1_scene.xml)
----
* 桌面(work_table): x∈[0.3, 1.3], y∈[-0.5, 0.5], 顶面 z=0.641, 方块静置中心 z=0.661
* 绿桶(green_bin): x∈[0.65, 0.75], y∈[-0.05, 0.05](含壁厚)。带内离桶最近的点
  (0.65, ±0.15) 与桶壁在 y 向仍有 8cm 间隙(x 向虽有重叠但 y 不相交,无碰撞)
* v3 老固定点 (0.65, ±0.30) 落在带的外后角 ⇒ v4 训练分布**包含** v3 的全部几何,
  --spawn home 评测在分布内,新老结果可直接对比
* y 取 ±[0.15, 0.30] ⇒ 两块中心距按构造 ≥0.30m,双臂互不打架

依赖仅 numpy + mujoco:数采(xr-robotics 环境)与 rollout(lingbotv2 环境)都能 import。
要调带只改 SPAWN_BAND,两边自动同步;改后务必重验 min_separation 与桶/桌边界。
"""
from __future__ import annotations

import mujoco
import numpy as np

CUBE_NAMES = ("red_cube_left", "red_cube_right")

# v4 随机出生带(基座系,单位米):x=前向, y=左正。【备用】实测遥操作对随机落点
# 舒适度不一, v4 正式采集改回固定点(见下);带与采样函数保留, 供后续版本启用。
SPAWN_BAND = {
    "x": (0.45, 0.65),           # 前后范围(0.65=v3 老固定点的前后坐标)
    "y_left": (+0.15, +0.30),    # 左臂方块(基座左侧,0.30=v3 老固定点)
    "y_right": (-0.30, -0.15),   # 右臂方块(镜像)
    "z": 0.741,                  # 与 home keyframe 同高,落稳交给 settle 静置
    "min_separation": 0.30,      # 两块中心最小间距(y 按构造保证,断言兜底防改带改坏)
}

# 固定出生点(基座系, z 与 keyframe 同高)。
# v4 固定点 = 与 v3 相同的老点 (0.65, ±0.30)——曾试过挪到双臂之间 (0.55, ±0.12),
# 后按用户要求改回 v3 老点,与历史数据/评测直接可比。定义在 H1_scene.xml 的
# home keyframe 里 —— A 键重置 / rollout --spawn home / 离线校验全部自动同源。
# 这里登记一份显式坐标(与 keyframe 数值一致):
FIXED_SPAWNS = {
    "v3": {   # v1~v4 一直沿用的固定点, 训练/评测历史数据都在这个几何上
        "red_cube_left": (0.65, +0.30, 0.741),
        "red_cube_right": (0.65, -0.30, 0.741),
    },
}

_rng = np.random.default_rng()


def _write_freejoint(model: mujoco.MjModel, data: mujoco.MjData,
                     name: str, pos) -> None:
    """把一个方块 freejoint 写为给定位置 + 单位四元数, 速度/加速度/外力清零。"""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if joint_id < 0 or body_id < 0:
        raise ValueError(f"模型缺少方块 freejoint/body: '{name}'")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)   # wxyz 单位四元数
    data.qvel[dof_adr:dof_adr + 6] = 0.0
    data.qacc[dof_adr:dof_adr + 6] = 0.0
    data.qacc_warmstart[dof_adr:dof_adr + 6] = 0.0
    data.xfrc_applied[body_id] = 0.0


def sample_cube_spawns(rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """【备用】按 SPAWN_BAND 采样一次出生位置,返回 {cube_name: np.array([x, y, z])}。"""
    r = rng if rng is not None else _rng
    spawns = {
        "red_cube_left": np.array([r.uniform(*SPAWN_BAND["x"]),
                                   r.uniform(*SPAWN_BAND["y_left"]),
                                   SPAWN_BAND["z"]]),
        "red_cube_right": np.array([r.uniform(*SPAWN_BAND["x"]),
                                    r.uniform(*SPAWN_BAND["y_right"]),
                                    SPAWN_BAND["z"]]),
    }
    sep = float(np.linalg.norm(spawns["red_cube_left"][:2] - spawns["red_cube_right"][:2]))
    if sep < SPAWN_BAND["min_separation"] - 1e-6:
        raise AssertionError(
            f"采样间距 {sep:.3f}m < 下限 {SPAWN_BAND['min_separation']}m,检查 SPAWN_BAND 是否改坏")
    return spawns


def apply_cube_spawns(model: mujoco.MjModel, data: mujoco.MjData,
                      rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """【备用】采样并写入两个 freejoint。写完由调用方负责 mj_forward/静置。"""
    spawns = sample_cube_spawns(rng)
    for name, pos in spawns.items():
        _write_freejoint(model, data, name, pos)
    return spawns


def apply_fixed_spawns(model: mujoco.MjModel, data: mujoco.MjData, name: str = "v3") -> dict[str, np.ndarray]:
    """写入 FIXED_SPAWNS 里登记的固定出生点(如 v3 老点,用于对比评测)。"""
    if name not in FIXED_SPAWNS:
        raise ValueError(f"未登记的固定点: '{name}'(可用: {list(FIXED_SPAWNS)})")
    spawns = {cube: np.array(pos) for cube, pos in FIXED_SPAWNS[name].items()}
    for cube, pos in spawns.items():
        _write_freejoint(model, data, cube, pos)
    return spawns


def spawns_str(spawns: dict[str, np.ndarray]) -> str:
    """一行可读日志,数采与 rollout 统一格式,方便事后核对出生点。"""
    return "  ".join(f"{name.replace('red_cube_', '')}: ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
                     for name, p in spawns.items())
