#!/usr/bin/env python3
"""Step 10: 键盘遥操作 —— 无 VR 硬件版, 复刻前辈 PICO 遥操作的软件骨架。

前辈链路:   PICO 头显+手柄 → WiFi → PC Service → Python SDK → IK → MuJoCo
本脚本链路: 键盘 (viewer 窗口聚焦时) → 同一套 IK → 同一个 MuJoCo 模型

刻意保留的前辈设计 (h1_pico_teleop.py):
  ① clutch 式相对控制: 按 N/M "激活"左右手臂时, 以当前 TCP 位姿为锚点,
     之后的目标 = 锚点 + 键盘累计增量; 冻结时手臂停在世界系当前位置
  ② 安全限位: 工作空间盒 + 距锚点最大平移 0.5m / 最大旋转 150°
  ③ IK 任务配方: posture(soft) 钉零空间 + trunk(hard) 钉腰头 + 每臂一个
     frame task (激活) 或 arm_hold 关节任务 (冻结) —— 权重随激活切换
  ④ 位置执行器: placo 解出的关节角直接写 ctrl (名字 {joint}_position)

键位 (点一下动一步: 平移 2cm / 旋转 10° / 头 0.15rad):
  1 / 2      切换 左/右臂 激活↔冻结      W/S  前进/后退 (±x)
  A/D        左/右平移 (±y)             R/F  升/降 (±z)
  Q/E        绕竖直轴 z 旋转            Z/C  绕前向轴 x 旋转
  G          夹爪 开↔合 (激活臂)        ←/→  头 yaw
  ↑/↓        头 pitch                   B    全身回 home
  H          帮助

用法:
  python scripts/10_keyboard_teleop.py             # 交互 (需要显示)
  python scripts/10_keyboard_teleop.py --check     # 无头自测 (CI/远程)
"""

import argparse
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import placo

HERE = Path(__file__).resolve().parent.parent
MJCF = HERE / "mujoco/H1_scene.xml"
IK_URDF = HERE / "urdf/H1_ik.urdf"

SIDES = ("left", "right")
# ── 抄自前辈的安全限位 ─────────────────────────────────
WORKSPACE_MIN = np.array([-0.05, -0.80, 0.40])
WORKSPACE_MAX = np.array([1.00, 0.80, 1.50])
MAX_TRANSLATION_FROM_ANCHOR = 0.50          # m
MAX_ROTATION_FROM_ANCHOR = np.radians(150)  # rad
# ── 本脚本的步进量 ─────────────────────────────────────
STEP_POS = 0.02      # 每键平移 2cm
STEP_ROT = np.radians(10)
STEP_HEAD = 0.15
CTRL_RATE = 3.0      # ctrl 斜坡限速 rad/s (防跳变, 见 Step 6 forcerange 教训)
SOLVE_EVERY = 5      # 每 5 个物理步解一次 IK → 100 Hz (timestep 0.002)

ARM_JOINTS = {
    "left": [f"Left_J{i}" for i in range(1, 8)],
    "right": ["Right_J1", "Right_J2", "Right_J3", "Right_J4",
              "Right_J5", "Right_j6", "Right_J7"],  # 注意 J6 是小写 j (URDF 原文如此)
}
TRUNK_JOINTS = ("waist_J1", "waist_J2", "head_j1", "Head_j2")

HELP_TEXT = """[H1 键盘遥操作] N/M=激活/冻结 左/右臂  WASD RF=平移  QE ZC=旋转
G=夹爪  J/L=头左右  I/K=头上下  B=回 home
V=相机复位  C=相机看门狗  P=状态快照  H=帮助
重要: 不要用数字键! viewer 内部占用数字键 (会让模型"消失"/幽灵渲染)"""


def glfw_key(code: int) -> str:
    """把 viewer 回调的键码翻译成符号 (字母/数字取 ASCII, 方向键取 GLFW 码)。"""
    special = {262: "RIGHT", 263: "LEFT", 264: "DOWN", 265: "UP"}
    if code in special:
        return special[code]
    if 32 <= code < 127:
        return chr(code)
    return f"KEY{code}"


class KeyboardTeleop:
    """键盘 → placo IK → MuJoCo ctrl, 一物三职的控制器。"""

    def __init__(self) -> None:
        # ── MuJoCo 侧 ──────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(
            self.model, self.data,
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        self.home_ctrl = self.data.ctrl.copy()

        # 关节 → 位置执行器/地址映射 (名字对名字, 不数下标)
        self.actuators, self.qposadr = {}, {}
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"{name}_position")
            if aid >= 0:  # 有位置执行器的才是 IK 关心的关节 (34 关节中的上身 18 个)
                self.actuators[name] = aid
                self.qposadr[name] = self.model.jnt_qposadr[j]
        self.gripper_act = {
            side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"{side}_omnipicker_gripper_opening")
            for side in SIDES}
        self.tcp_site = {
            side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                    f"{side}_omnipicker_tcp")
            for side in SIDES}

        # ── placo 侧 (配方与 Step 9 自测 B 相同) ────────
        self.robot = placo.RobotWrapper(str(IK_URDF))
        self.home = {n: float(self.data.qpos[a]) for n, a in self.qposadr.items()}
        for n, v in self.home.items():
            self.robot.set_joint(n, v)
        self.robot.update_kinematics()

        self.solver = self.robot.make_solver()
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)

        # posture: 钉住全部 18 关节在 home 附近 —— 没有它零空间会漂到限位
        self.posture = self.solver.add_joints_task()
        self.posture.set_joints(self.home)
        self.posture.configure("posture", "soft", 1e-3)

        # trunk: 腰钉死 home, 头两关节由方向键驱动
        self.trunk_target = {n: self.home[n] for n in TRUNK_JOINTS}
        self.trunk = self.solver.add_joints_task()
        self.trunk.set_joints(self.trunk_target)
        self.trunk.configure("trunk", "hard", 1.0)

        # 每臂一对任务: 激活时 frame task 高权重; 冻结时关节空间钉住
        self.frame_task, self.hold_task = {}, {}
        for side in SIDES:
            link = f"{side}_omnipicker_tcp_link"
            self.frame_task[side] = self.solver.add_frame_task(
                link, self.robot.get_T_world_frame(link))
            self.frame_task[side].configure(f"{side}_tcp", "soft", 0.0)
            self.hold_task[side] = self.solver.add_joints_task()
            self.hold_task[side].set_joints(
                {j: self.home[j] for j in ARM_JOINTS[side]})
            self.hold_task[side].configure(f"{side}_hold", "soft", 1.0)

        self.active = {"left": False, "right": False}
        self.anchor = {}                     # 激活瞬间的 TCP (pos + quat)
        self.target_pos, self.target_quat = {}, {}
        self.gripper_cmd = {side: float(self.home_ctrl[aid])
                            for side, aid in self.gripper_act.items()}
        self.key_queue: deque[str] = deque()  # GUI 线程 → 控制线程的通道
        self.reset_cam_fn = None              # run_viewer 注入: V 键复位相机
        self.cam_lock = True                  # 相机看门狗: 锁定自由视角
        self._cam = None                      # run_viewer 注入: viewer.cam 引用
        # 方块等 freejoint body (取证快照用: 方块飞走会让画面看起来异常)
        self._cube_bodies = []
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            for j in range(self.model.body_jntnum[b]):
                if self.model.jnt_type[self.model.body_jntadr[b] + j] == \
                        mujoco.mjtJoint.mjJNT_FREE:
                    self._cube_bodies.append((name, b))

    # ── 激活 / 冻结 (clutch) ───────────────────────────
    def toggle_arm(self, side: str) -> None:
        if not self.active[side]:
            # 踩坑实录: mj_resetDataKeyframe/mj_step 之前 site_xpos 是全 0,
            # 此刻激活会把锚点设到世界原点 -> QP 求解直接 NaN 崩溃
            # (用户看到的现象: 按 1 后 viewer 窗口整个消失)。
            # 修法: 读 site 前强制把运动学算出来, 并校验锚点有限且在工作空间内
            mujoco.mj_forward(self.model, self.data)
            T = np.eye(4)
            pos = self.data.site_xpos[self.tcp_site[side]].copy()
            if (not np.isfinite(pos).all()
                    or (pos < WORKSPACE_MIN - 0.05).any()
                    or (pos > WORKSPACE_MAX + 0.05).any()):
                print(f"  [{side} 臂] 激活失败: 当前 TCP 位姿异常 {np.round(pos, 3)}, "
                      "已忽略 (物理可能未初始化)")
                return
            T[:3, 3] = pos
            R = self.data.site_xmat[self.tcp_site[side]].reshape(3, 3)
            T[:3, :3] = R
            self.anchor[side] = T
            self.target_pos[side] = T[:3, 3].copy()
            self.target_quat[side] = self._quat(R)
            self.frame_task[side].T_world_frame = T
            self._set_weights(side, tcp=True)
            self.active[side] = True
            print(f"  [{side} 臂] 激活, 锚点 {np.round(T[:3, 3], 3)}")
            self.dump_render_state("激活")   # 自动取证: 消失瞬间的渲染状态
        else:
            # 冻结: 把该臂钉在"此刻"的关节角上 (世界系冻结, 不回 home)
            self.hold_task[side].set_joints(
                {j: float(self.robot.get_joint(j)) for j in ARM_JOINTS[side]})
            self._set_weights(side, tcp=False)
            self.active[side] = False
            print(f"  [{side} 臂] 冻结")

    def _set_weights(self, side: str, tcp: bool) -> None:
        """激活 → frame task 接管; 冻结 → 关节空间保持 (前辈 _configure_arm_tasks)。"""
        self.frame_task[side].configure(f"{side}_tcp", "soft", 1.0 if tcp else 0.0)
        self.hold_task[side].configure(f"{side}_hold", "soft", 0.0 if tcp else 1.0)

    # ── 目标更新 (带全部安全限位) ──────────────────────
    def move_tcp(self, side: str, dpos: np.ndarray, rot_axis: str,
                 dangle: float) -> None:
        if not self.active[side]:
            print(f"  [{side} 臂] 未激活, 先按 {'N' if side == 'left' else 'M'}")
            return
        new_pos = self.target_pos[side] + dpos
        # 限位 ①: 工作空间盒  ②: 距锚点 ≤ 0.5m
        new_pos = np.clip(new_pos, WORKSPACE_MIN, WORKSPACE_MAX)
        if np.linalg.norm(new_pos - self.anchor[side][:3, 3]) > \
                MAX_TRANSLATION_FROM_ANCHOR:
            print("  [限位] 距锚点超 0.5m, 拒绝")
            new_pos = self.target_pos[side]
        self.target_pos[side] = new_pos

        new_quat = self.target_quat[side]
        if dangle != 0.0:
            axis = {"z": (0, 0, 1), "x": (1, 0, 0)}[rot_axis]
            dq = self._axis_quat(axis, dangle)
            new_quat = self._quat_mul(dq, self.target_quat[side])
            # 限位 ③: 相对锚点总旋转 ≤ 150°
            ra = self._quat(self.anchor[side][:3, :3])
            if self._quat_angle(ra, new_quat) > MAX_ROTATION_FROM_ANCHOR:
                print("  [限位] 相对锚点旋转超 150°, 拒绝")
                new_quat = self.target_quat[side]
        self.target_quat[side] = new_quat

        T = np.eye(4)
        T[:3, :3] = self._rotmat(new_quat)
        T[:3, 3] = self.target_pos[side]
        self.frame_task[side].T_world_frame = T

    # ── 按键处理 ──────────────────────────────────────
    def on_key(self, keycode: int) -> None:
        """viewer 的 GUI 线程调用 —— 只入队, 不碰求解器。"""
        self.key_queue.append(glfw_key(keycode))

    def drain_keys(self) -> None:
        while self.key_queue:
            k = self.key_queue.popleft()
            moves = {"W": ((0.02, 0, 0), 0), "S": ((-0.02, 0, 0), 0),
                     "A": ((0, 0.02, 0), 0), "D": ((0, -0.02, 0), 0),
                     "R": ((0, 0, 0.02), 0), "F": ((0, 0, -0.02), 0)}
            act = [s for s in SIDES if self.active[s]]
            arm_key = k in moves or k in ("Q", "E", "Z", "C") or k == "G"
            if arm_key and not act:
                # 不再静默忽略 —— 没激活时按 WASD 等键必须给出可见提示
                print("  [提示] 手臂未激活: 先按 N(左臂) / M(右臂) 激活, "
                      "再用 WASD·RF·QE·ZC·G 控制")
            if k == "N":
                self.toggle_arm("left")
            elif k == "M":
                self.toggle_arm("right")
            elif k in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0"):
                # 隔离测试实锤: viewer 的 C++ 层内部占用数字键 (帮助表里没写),
                # 按 2 会让模型"消失"、按 3 出现幽灵渲染 —— 一律拦下并提示
                print("  [提示] 数字键被 viewer 内部占用, 请用 N(左臂)/M(右臂)")
            elif k in moves:
                d, _ = moves[k]
                for side in act:
                    self.move_tcp(side, np.array(d), "z", 0.0)
            elif k in ("Q", "E", "Z", "C"):
                ang = {"Q": STEP_ROT, "E": -STEP_ROT,
                       "Z": STEP_ROT, "C": -STEP_ROT}[k]
                axis = "z" if k in ("Q", "E") else "x"
                for side in act:
                    self.move_tcp(side, np.zeros(3), axis, ang)
            elif k == "G":
                for side in act:
                    self.gripper_cmd[side] = 1.0 - self.gripper_cmd[side]
                    print(f"  [{side} 夹爪] "
                          f"{'开' if self.gripper_cmd[side] > 0.5 else '合'}")
            elif k in ("J", "L", "I", "K"):
                # 头部控制。绝不能用方向键: viewer 内置绑定 ←→=切换相机,
                # 会切到头戴/腕部相机, 视角变成"从机器人眼里往外看",
                # 用户以为模型消失了 (v1.2 之前的真实踩坑)
                d = {"J": -1, "L": 1, "I": 1, "K": -1}[k] * STEP_HEAD
                j = "head_j1" if k in ("J", "L") else "Head_j2"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                self.trunk_target[j] = float(np.clip(
                    self.trunk_target[j] + d, *self.model.jnt_range[jid]))
                self.trunk.set_joints(self.trunk_target)
            elif k in ("LEFT", "RIGHT", "UP", "DOWN"):
                print("  [提示] 方向键未绑定头部 (viewer 内置键: [ ]=切相机, Esc=自由相机)。"
                      "头部请用 J/L(左右) I/K(上下); 视角乱了按 V 复位")
            elif k == "V":
                if self.reset_cam_fn is not None:
                    self.reset_cam_fn()
                    print("  [相机] 已复位到默认自由视角")
            elif k == "C":
                self.cam_lock = not self.cam_lock
                print(f"  [相机] 看门狗{'开启 (自动弹回自由视角)' if self.cam_lock else '关闭 (可任意切相机)'}")
            elif k == "P":
                self.dump_render_state("手动P键")
            elif k == "B":
                self.go_home()
            elif k == "H":
                print(HELP_TEXT)

    def go_home(self) -> None:
        """双臂冻结回 home 姿态, 头腰复位, 夹爪回 home 开度。"""
        for side in SIDES:
            if self.active[side]:
                self.toggle_arm(side)
            self.hold_task[side].set_joints(
                {j: self.home[j] for j in ARM_JOINTS[side]})
            self.gripper_cmd[side] = float(
                self.home_ctrl[self.gripper_act[side]])
        self.trunk_target = {n: self.home[n] for n in TRUNK_JOINTS}
        self.trunk.set_joints(self.trunk_target)
        print("  [home] 全身复位中 (ctrl 斜坡过渡)")

    # ── 主循环一步: IK → ctrl 斜坡 → 物理 ─────────────
    def step(self, steps: int = SOLVE_EVERY) -> None:
        for i in range(steps):
            if i == 0:
                self.drain_keys()
                try:
                    self.solver.solve(True)   # 每个控制周期一次 (前辈节奏)
                except RuntimeError as e:
                    # QP 偶发 NaN (目标不可达/奇异位形) 时自愈而不是窗口消失:
                    # 把 placo 状态拉回真实物理状态, 任务目标重锚到当前 TCP
                    print(f"  [IK 异常, 自愈] {e}")
                    self._resync_placo_from_mujoco()
                else:
                    if not np.isfinite(self.robot.state.q).all():
                        # 静默 NaN 防御: solve 也可能"返回"NaN 而不抛异常,
                        # 不检查的话 NaN 会无声灌进 ctrl -> qpos -> 模型消失
                        print("  [IK 输出含 NaN, 自愈] 目标不可达或奇异位形")
                        self._resync_placo_from_mujoco()
                    else:
                        self.robot.update_kinematics()  # 不刷新雅可比会过期
            # 关节角 → 目标 ctrl, 带斜坡限速防跳变; 非有限值一律跳过
            for name, aid in self.actuators.items():
                target = float(self.robot.get_joint(name))
                if not np.isfinite(target):
                    continue
                cur = self.data.ctrl[aid]
                max_d = CTRL_RATE * self.model.opt.timestep
                self.data.ctrl[aid] = cur + np.clip(target - cur, -max_d, max_d)
            for side, aid in self.gripper_act.items():
                self.data.ctrl[aid] = self.gripper_cmd[side]
            mujoco.mj_step(self.model, self.data)

    def _resync_placo_from_mujoco(self) -> None:
        """IK 崩溃后的自愈: placo 状态 ← MuJoCo 真实关节角, 目标 ← 当前 TCP。

        不自愈的话 placo 里留着发散解, 每个 ctrl 写入都会把 NaN 灌进物理。
        """
        mujoco.mj_forward(self.model, self.data)
        for name, adr in self.qposadr.items():
            self.robot.set_joint(name, float(self.data.qpos[adr]))
        self.robot.update_kinematics()
        for side in SIDES:
            if self.active[side]:
                T = np.eye(4)
                T[:3, 3] = self.data.site_xpos[self.tcp_site[side]]
                T[:3, :3] = self.data.site_xmat[self.tcp_site[side]].reshape(3, 3)
                self.anchor[side] = T
                self.target_pos[side] = T[:3, 3].copy()
                self.target_quat[side] = self._quat(T[:3, :3])
                self.frame_task[side].T_world_frame = T
        self.posture.set_joints(
            {n: float(self.data.qpos[a]) for n, a in self.qposadr.items()})

    def dump_render_state(self, tag: str) -> None:
        """取证快照: "模型消失"时按 P, 把相机/场景/物理状态全部打出来。

        相机数值正常却看不见模型 → 渲染层问题; 相机数值离谱 → 相机问题;
        物理有 inf/方块飞走 → 物理问题。三类原因一网打尽。
        """
        if self.reset_cam_fn is None:      # 无头/自测模式下没有 viewer
            return
        cam = self._cam
        q = self.data.qpos
        cubes = [(n, np.round(self.data.xpos[b], 1).tolist())
                 for n, b in self._cube_bodies]
        print(f"  [快照·{tag}] cam.type={int(cam.type)} fixedcamid={cam.fixedcamid} "
              f"lookat={np.round(cam.lookat, 2).tolist()} dist={cam.distance:.2f} "
              f"azim={cam.azimuth:.0f} elev={cam.elevation:.0f} | "
              f"qpos有限={np.isfinite(q).all()} 范围=[{q.min():.2f},{q.max():.2f}] | "
              f"方块={cubes}")

    # ── 四元数小工具 (wxyz 顺序, pinocchio 惯例) ──────
    @staticmethod
    def _quat(R: np.ndarray) -> np.ndarray:
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.reshape(9))
        return q

    @staticmethod
    def _rotmat(q: np.ndarray) -> np.ndarray:
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3)

    @staticmethod
    def _axis_quat(axis, angle: float) -> np.ndarray:
        q = np.array([np.cos(angle / 2),
                      *(np.sin(angle / 2) * np.array(axis[:3]))])
        return q

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.empty(4)
        mujoco.mju_mulQuat(out, a, b)
        return out

    @staticmethod
    def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
        return 2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))


def run_viewer() -> None:
    """交互模式: viewer 窗口接收按键, 控制线程跑物理 (对表 1x 实时)。"""
    import time
    import traceback

    teleop = KeyboardTeleop()
    print(HELP_TEXT)
    print("版本: v1.6 激活键改 N/M —— 数字键被 viewer C++ 层内部占用 (会让模型消失)")
    home_id = mujoco.mj_name2id(teleop.model, mujoco.mjtObj.mjOBJ_KEY, "home")

    def hard_recover() -> None:
        """物理层兜底: 复位 home + IK 重锚 + 双臂回冻结态。"""
        mujoco.mj_resetDataKeyframe(teleop.model, teleop.data, home_id)
        mujoco.mj_forward(teleop.model, teleop.data)
        teleop._resync_placo_from_mujoco()
        for side in teleop.gripper_act:
            teleop.gripper_cmd[side] = float(
                teleop.home_ctrl[teleop.gripper_act[side]])
        for side in SIDES:
            teleop.active[side] = False
            teleop._set_weights(side, tcp=False)
            teleop.hold_task[side].set_joints(
                {j: teleop.home[j] for j in ARM_JOINTS[side]})

    dt = teleop.model.opt.timestep * SOLVE_EVERY   # 每轮 step() 的仿真时长
    next_t = time.perf_counter()
    with mujoco.viewer.launch_passive(teleop.model, teleop.data,
                                      key_callback=teleop.on_key) as viewer:
        teleop.reset_cam_fn = lambda: mujoco.mjv_defaultFreeCamera(
            teleop.model, viewer.cam)
        teleop._cam = viewer.cam
        while viewer.is_running():
            # 相机看门狗: 方向键会让 viewer 内部切到固定相机 (头/腕视角,
            # 看似"模型消失")。锁定时自动弹回自由视角并留下记录,
            # 若"消失"时这里没打印, 说明另有真凶 —— 日志会作证
            if teleop.cam_lock and viewer.cam.fixedcamid != -1:
                mujoco.mjv_defaultFreeCamera(teleop.model, viewer.cam)
                print("  [相机看门狗] 检测到切到固定相机(方向键副作用), 已弹回自由视角 (C 可解锁)")
            try:
                # 看门狗: 物理 NaN/inf (无论来源) -> 复位, 模型永不消失
                if not np.isfinite(teleop.data.qpos).all():
                    print("  [物理 NaN/inf, 复位 home]")
                    hard_recover()
                teleop.step()
            except Exception:
                # 任何漏网异常: 记栈 + 兜底恢复, 窗口绝不退出
                
                tb = traceback.format_exc()
                print("  [未预期异常, 已恢复, 栈见 /tmp/teleop_crash.log]")
                print(tb)
                with open("/tmp/teleop_crash.log", "w") as f:
                    f.write(tb)
                hard_recover()
            viewer.sync()
            # 对表实时: 裸循环会跑到几十倍速, 遥操作手感会乱
            next_t += dt
            pause = next_t - time.perf_counter()
            if pause > 0:
                time.sleep(pause)
            else:
                next_t = time.perf_counter()   # 落后太多 (卡顿/调试断点) 就重新对表
            # 对表实时: 裸循环会跑到几十倍速, 遥操作手感会乱
            next_t += dt
            pause = next_t - time.perf_counter()
            if pause > 0:
                time.sleep(pause)
            else:
                next_t = time.perf_counter()   # 落后太多 (卡顿/调试断点) 就重新对表
    print("遥操作结束")


def run_check() -> None:
    """无头自测: 注入按键序列, 断言 IK/限位/夹爪/回 home 全链路正确。"""
    teleop = KeyboardTeleop()
    mujoco.mj_forward(teleop.model, teleop.data)
    p0 = teleop.data.site_xpos[teleop.tcp_site["left"]].copy()

    print("── check 1: 激活左臂 + W×5 (期望 +10cm, clamp 后到位) ──")
    teleop.key_queue.extend(["N"] + ["W"] * 5)
    for _ in range(1000):
        teleop.step()
    p1 = teleop.data.site_xpos[teleop.tcp_site["left"]]
    dx = p1[0] - p0[0]
    print(f"  实际 x 位移 {dx * 100:.1f} cm")
    assert 0.06 < dx < 0.14, "左臂前伸未达预期"

    print("── check 2: 限位 —— W×100 (期望目标被钉在 x ≤ min(1.0, 锚+0.5)) ──")
    teleop.key_queue.extend(["W"] * 100)
    for _ in range(500):
        teleop.step()
    px = teleop.target_pos["left"][0]
    print(f"  目标 x = {px:.3f} (锚点 x={teleop.anchor['left'][0, 3]:.3f}+0.5)")
    assert px <= min(WORKSPACE_MAX[0], teleop.anchor["left"][0, 3] + 0.51) + 1e-6

    print("── check 3: 夹爪 G (期望 ctrl 翻转且手指闭合) ──")
    g0 = teleop.data.ctrl[teleop.gripper_act["left"]]
    teleop.key_queue.append("G")
    for _ in range(2000):
        teleop.step()
    gid = mujoco.mj_name2id(teleop.model, mujoco.mjtObj.mjOBJ_JOINT,
                            "left_omnipicker_hand_narrow1_joint")
    q_narrow = teleop.data.qpos[teleop.model.jnt_qposadr[gid]]
    print(f"  ctrl {g0:.1f}→{teleop.data.ctrl[teleop.gripper_act['left']]:.1f}, "
          f"narrow1={q_narrow:.2f} rad")
    assert teleop.data.ctrl[teleop.gripper_act["left"]] != g0

    print("── check 4: B 回 home (期望上身 18 关节回 home ±0.1 rad) ──")
    teleop.key_queue.append("B")
    for _ in range(4000):
        teleop.step()
    worst = max(
        abs(teleop.data.qpos[a] - teleop.home[n]) for n, a in teleop.qposadr.items())
    print(f"  最大关节偏差 {worst:.3f} rad")
    assert worst < 0.1, "未回 home"

    print("\n=== Step 10 自测全部通过 ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="无头自测, 不开窗口")
    args = parser.parse_args()
    run_check() if args.check else run_viewer()


if __name__ == "__main__":
    main()
