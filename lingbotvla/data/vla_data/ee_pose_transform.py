# -*- coding: utf-8 -*-
"""
ee_pose_transform.py —— 末端位姿(end.position)的相对位姿变换
====================================================================
【背景】末端执行器位姿 = 位置(xyz,3维)+ 旋转(四元数 xyzw,4维)= 7维/臂(pose_dim=7)。
       双臂时拼接成 14 维。

【为什么需要单独处理】当 action.end.position 设 subtract_state=True(学相对动作)时:
       · 位置可以直接相减:rel_xyz = action_xyz - state_xyz  ✅
       · 但旋转(四元数)不能直接相减!四元数空间不是线性的,q1-q2 没有几何意义  ❌
       正确做法用四元数乘法求"相对旋转":rel_q = state_q⁻¹ ⊗ action_q

【两个核心函数】(互为逆运算)
  relative_pose_quaternion  绝对位姿 → 相对位姿(训练 apply 用,模型学相对增量)
  absolute_pose_quaternion  相对位姿 → 绝对位姿(推理 unapply 用,还原成机器人可执行的位姿)

【注意】本文件四元数统一用 xyzw 格式(x,y,z 虚部在前,w 标量在后)。
【被谁调用】utils.py 的 FeatureTransform.apply / unapply(仅 end.position + subtract_state=True 时)。
====================================================================
"""

import torch
import numpy as np
from torch import Tensor

__all__ = [                       # 只对外暴露这 3 个(工具函数是内部用的)
    '_is_quaternion_relative_type',
    'relative_pose_quaternion',
    'absolute_pose_quaternion',
]

# ============== 四元数基础工具(xyzw 格式)==============
# 四元数 q=(x,y,z,w),w 是标量,(x,y,z) 是虚部。单位四元数表示一个 3D 旋转。

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion in xyzw format. Shape: (..., 4).
    归一化为单位四元数(只有单位四元数才表示纯旋转)。"""
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / norm


def quat_canonicalize(q: torch.Tensor) -> torch.Tensor:
    """Flip quaternion sign so the scalar part keeps a stable non-negative sign.
    规范化符号:让 w(标量)保持非负。
    因为 q 和 -q 表示同一个旋转,统一符号避免这种"双义性"干扰模型学习。"""
    sign = torch.where(q[..., 3:4] < 0, -1.0, 1.0)
    return q * sign


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Inverse of quaternion in xyzw format. Shape: (..., 4).
    求逆。单位四元数的逆 = 共轭 = 虚部取反(x,y,z 取负,w 不变)。"""
    inv = quat_normalize(q).clone()
    inv[..., :3] = -inv[..., :3]  # negate xyz (imaginary part)  虚部取反
    return inv


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2 in xyzw format. Shape: (..., 4)
    四元数乘法(Hamilton 积)。几何意义:旋转的复合——q1⊗q2 表示"先转 q2 再转 q1"。"""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector(s) by quaternion(s) in xyzw format.
    用四元数 q 旋转向量 v(把 v 转到 q 代表的旋转下)。"""
    q = quat_normalize(q)
    v_quat = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)   # 向量补成纯四元数 (v, 0)
    return quat_multiply(quat_multiply(q, v_quat), quat_inverse(q))[..., :3]   # q ⊗ v ⊗ q⁻¹


def quat_rotate_inverse(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vector(s) by the inverse of quaternion(s) in xyzw format.
    用 q 的逆旋转向量(把 v 从 q 的旋转下转回去)。"""
    return quat_rotate(quat_inverse(q), v)


def _resolve_quaternion_relative_type(relative_type: str | None) -> str:
    """把各种写法统一成 'world' 或 'local' 两种。
    world:相对位置差在世界坐标系;local:相对位置差转到当前位姿的局部坐标系。"""
    if relative_type in (None, 'world', 'quaternion', 'quaternion_world'):
        return 'world'
    if relative_type in ('local', 'quaternion_local'):
        return 'local'
    raise ValueError(f"Unsupported quaternion relative type: {relative_type}")


def _is_quaternion_relative_type(relative_type: str | None) -> bool:
    """判断是否走"四元数相对位姿"分支(只对 end.position 这类有旋转的特征有意义)。"""
    return relative_type in {'quaternion', 'quaternion_world', 'quaternion_local'}



def relative_pose_quaternion(
    action: torch.Tensor,
    state: torch.Tensor,
    pose_dim: int = 7,
    relative_type: str | None = 'world',
) -> torch.Tensor:
    """★ 训练用(apply):把"绝对位姿 action"转成"相对当前 state 的相对位姿"。
    支持 dual-arm:数据按 pose_dim(=7)切块,每块一个臂独立处理。
    action/state shape: (..., N*pose_dim),N 是臂数。
    """
    relative_type = _resolve_quaternion_relative_type(relative_type)
    total_dim = action.shape[-1]
    assert total_dim % pose_dim == 0          # 必须是 7 的整数倍(每臂 3 位置 + 4 四元数)
    chunks = total_dim // pose_dim             # 臂数
    parts = []
    for i in range(chunks):
        s = i * pose_dim
        a_xyz = action[..., s:s+3]                       # 动作位置 xyz
        a_q = quat_normalize(action[..., s+3:s+pose_dim])  # 动作旋转四元数
        s_xyz = state[..., s:s+3]                        # 当前状态位置 xyz
        s_q = quat_normalize(state[..., s+3:s+pose_dim])   # 当前状态旋转四元数
        # ---- 位置:相对量 ----
        if relative_type == 'local':
            # local:把位置差用 state 旋转的"逆"转到当前位姿的局部坐标系
            rel_xyz = quat_rotate_inverse(a_xyz - s_xyz, s_q)
        else:
            # world:位置差直接相减(世界坐标系)
            rel_xyz = a_xyz - s_xyz
        # ---- 旋转:相对旋转 = state 的逆 ⊗ action ----
        # (几何:从 state 朝向到 action 朝向,需要转的那个相对旋转)
        rel_q = quat_canonicalize(quat_normalize(quat_multiply(quat_inverse(s_q), a_q)))
        parts.append(torch.cat([rel_xyz, rel_q], dim=-1))
    return torch.cat(parts, dim=-1)


def absolute_pose_quaternion(
    rel_action: torch.Tensor,
    state: torch.Tensor,
    pose_dim: int = 7,
    relative_type: str | None = 'world',
) -> torch.Tensor:
    """Recover absolute pose from relative pose (inverse of relative_pose_quaternion).
    ★ 推理用(unapply):模型预测出"相对位姿",加上当前 state 还原成"绝对位姿"(机器人执行)。
    rel_action/state shape: (..., N*pose_dim).
    """
    relative_type = _resolve_quaternion_relative_type(relative_type)
    total_dim = rel_action.shape[-1]
    assert total_dim % pose_dim == 0
    chunks = total_dim // pose_dim
    parts = []
    for i in range(chunks):
        s = i * pose_dim
        r_xyz = rel_action[..., s:s+3]                   # 相对位置
        r_q = quat_normalize(rel_action[..., s+3:s+pose_dim])  # 相对旋转
        s_xyz = state[..., s:s+3]
        s_q = quat_normalize(state[..., s+3:s+pose_dim])
        # ---- 位置:绝对 = state + 相对 ----
        if relative_type == 'local':
            # local:先把相对位置(在局部系)用 state 旋转转回世界系,再加 state 位置
            abs_xyz = s_xyz + quat_rotate(s_q, r_xyz)
        else:
            abs_xyz = s_xyz + r_xyz
        # ---- 旋转:绝对旋转 = state ⊗ 相对旋转 ----
        abs_q = quat_canonicalize(quat_normalize(quat_multiply(s_q, r_q)))
        parts.append(torch.cat([abs_xyz, abs_q], dim=-1))
    return torch.cat(parts, dim=-1)

def matrix_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix (..., 3, 3) to quaternion (..., 4) in xyzw format.
    旋转矩阵 → 四元数。

    Uses Shepperd's method for numerical stability.
    用 Shepperd 方法(根据 trace 和对角元素选最稳的分支,避免数值不稳定)。
    """
    batch_shape = R.shape[:-2]
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros(*batch_shape, 4, dtype=R.dtype, device=R.device)

    # 分支 1:trace > 0(通用情况)
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2
    q[..., 3] = 0.25 * s
    q[..., 0] = (m21 - m12) / s
    q[..., 1] = (m02 - m20) / s
    q[..., 2] = (m10 - m01) / s

    # 分支 2:m00 最大且 trace<=0(换公式避免除以接近 0 的 s)
    cond2 = (m00 > m11) & (m00 > m22) & (trace <= 0)
    s2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond2, 0.25 * s2, q[..., 0])
    q[..., 1] = torch.where(cond2, (m01 + m10) / s2, q[..., 1])
    q[..., 2] = torch.where(cond2, (m02 + m20) / s2, q[..., 2])
    q[..., 3] = torch.where(cond2, (m21 - m12) / s2, q[..., 3])

    # 分支 3:m11 最大
    cond3 = (~cond2) & (m11 > m22) & (trace <= 0)
    s3 = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond3, (m01 + m10) / s3, q[..., 0])
    q[..., 1] = torch.where(cond3, 0.25 * s3, q[..., 1])
    q[..., 2] = torch.where(cond3, (m12 + m21) / s3, q[..., 2])
    q[..., 3] = torch.where(cond3, (m02 - m20) / s3, q[..., 3])

    # 分支 4:m22 最大
    cond4 = (~cond2) & (~cond3) & (trace <= 0)
    s4 = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-10)) * 2
    q[..., 0] = torch.where(cond4, (m02 + m20) / s4, q[..., 0])
    q[..., 1] = torch.where(cond4, (m12 + m21) / s4, q[..., 1])
    q[..., 2] = torch.where(cond4, 0.25 * s4, q[..., 2])
    q[..., 3] = torch.where(cond4, (m10 - m01) / s4, q[..., 3])

    return quat_normalize(q)
