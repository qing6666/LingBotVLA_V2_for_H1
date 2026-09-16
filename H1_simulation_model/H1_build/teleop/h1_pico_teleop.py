#!/usr/bin/env python3
"""H1 + 双 OmniPicker 的 PICO 双臂遥操作主入口。

文件关系
--------
* ``mujoco/h1_omnipicker.xml``：动力学、任务场景、夹爪接触和三台相机。
* ``mujoco/h1_for_import.urdf``：Placo 逆运动学使用的运动学链。
* 本文件：读取 PICO 头显/双手柄/腰部 Tracker，求解 IK，向 MuJoCo 写入执行器目标，
  并显示头部主视角与左右腕部相机窗口。
* ``teleop/h1_dataset_recorder.py``：``--record`` 时把三相机图像与 state/action 写入
  LeRobotDataset；不依赖 XRoboToolkit，可用 ``teleop/h1_record_check.py`` 无头验证。

名称约束：XML 与 URDF 中的关节、TCP link 和 actuator 名称在这里被直接引用；修改
模型名称后，应先运行 ``python teleop/h1_pico_teleop.py --check-model``。

控制映射
--------
* 按住左/右 Grip：将该手柄与对应 TCP 对齐，并控制该机械臂运动。
* 松开 Grip：机械臂保持最后一个目标；再次按住可重新建立相对零点。
* 左/右 Trigger：控制对应 OmniPicker 闭合（0=张开，1=闭合）。
* 右手 B：解除两臂控制，使上半身平滑回到 ``home`` 姿态；录制中按下会先保存 episode。
* 右手 X：录制开关（需 ``--record``）——第一次按开始 episode，再按保存结束；
  录制中按 A 重置方块会先丢弃当前 episode（方块瞬移会污染数据）。

控制器刻意使用手柄的相对运动。因此标定时，机器人 TCP 不会跳到手柄的绝对世界坐标。

主 MuJoCo Viewer 默认使用 ``head_rgb``。左右腕部 RGB 图像默认显示在两个额外的
OpenCV 窗口中。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Dict

import cv2
import mujoco
from mujoco import viewer as mj_viewer
import numpy as np
import placo
from meshcat import transformations as tf

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.mujoco_utils import calc_placo_q_from_mujoco_qpos


H1_ROOT = Path(__file__).resolve().parents[1]
# 默认加载的动力学场景与 Placo IK 运动学模型。
DEFAULT_MJCF = H1_ROOT / "mujoco" / "H1_scene.xml"
DEFAULT_URDF = H1_ROOT / "urdf" / "H1_ik.urdf"
# --record 的默认数据集位置；与 h1_dataset_recorder.py 保持一致（这里不 import
# lerobot，因此 --check-model 在未安装 lerobot 的环境也能运行）。
# v3 = 夹爪斜坡版数采批次（与 v2 阶跃动作分布不同，禁止混批）
DEFAULT_REPO_ID = "mjq/h1_build_pick_v3"
DEFAULT_TASK = "Pick up the red cube and put it into the green bin"
DEFAULT_DATASET_DIR = H1_ROOT / "data" / DEFAULT_REPO_ID.split("/")[-1]
# 夹爪指令斜坡限速（行程/秒）：不限速时 trigger 一按到底，记录的 action 是 2 帧阶跃，
# 而夹爪物理闭合要 ~1.8s，阶跃形状模型学不动 → 推理时闭合迟、闭合浅、抓不住。
GRIPPER_COMMAND_SPEED = 2.0  # 全行程 ≈0.5s（约 15 帧@30fps），与物理跟进同量级、可学

SIDES = ("left", "right")
# 三台相机名称必须与 MJCF 中 <camera name="..."> 完全一致。
HEAD_CAMERA_NAME = "head_rgb"
# v4: 头部完全固定 —— 永远保持 home keyframe 姿态(head_j1=0、Head_j2=0.53 正对
# 桌面),头显不再联动头部(想恢复跟随改 False)。固定后头部相机成为几何常量的
# overlooking 相机:数据里 state[16:18] 为常量,模型无需学头部控制,回合间视角零散射。
HEAD_FIXED_AT_HOME = True
WRIST_CAMERA_NAMES = {
    "left": "left_wrist_rgb",
    "right": "right_wrist_rgb",
}
WRIST_VIEWER_TITLES = {
    "left": "H1 left wrist RGB",
    "right": "H1 right wrist RGB",
}
# 仅用于实时显示；后续采集数据时可在此处统一调整渲染分辨率和频率。
WRIST_CAMERA_WIDTH = 512
WRIST_CAMERA_HEIGHT = 512
WRIST_CAMERA_FPS = 30.0
ARM_JOINTS = {
    "left": ("Left_J1", "Left_J2", "Left_J3", "Left_J4", "Left_J5", "Left_J6", "Left_J7"),
    "right": ("Right_J1", "Right_J2", "Right_J3", "Right_J4", "Right_J5", "Right_j6", "Right_J7"),
}
DEFAULT_TRACKER_SERIAL = "PC2310MLL1091994G"
UPPER_BODY_JOINTS = (
    "waist_J1",
    "waist_J2",
    "Left_J1",
    "Left_J2",
    "Left_J3",
    "Left_J4",
    "Left_J5",
    "Left_J6",
    "Left_J7",
    "Right_J1",
    "Right_J2",
    "Right_J3",
    "Right_J4",
    "Right_J5",
    "Right_j6",
    "Right_J7",
    "head_j1",
    "Head_j2",
)


def manipulator_config() -> Dict[str, Dict[str, str]]:
    """返回左右臂的名称映射。

    ``link_name`` 必须是 URDF 和 MJCF 共有的 TCP body/link；其余字段是 XRoboToolkit
    中的手柄输入名、MuJoCo 夹爪 actuator 名，以及仅用于 Viewer 调试的目标 marker。
    """
    return {
        "left": {
            "link_name": "left_omnipicker_tcp_link",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_trigger": "left_trigger",
            "gripper_actuator": "left_omnipicker_gripper_opening",
            "vis_target": "left_teleop_target",
        },
        "right": {
            "link_name": "right_omnipicker_tcp_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "gripper_trigger": "right_trigger",
            "gripper_actuator": "right_omnipicker_gripper_opening",
            "vis_target": "right_teleop_target",
        },
    }


class H1PicoTeleopController(MujocoTeleopController):
    """H1 专用状态机，建立在 XRoboToolkit + Placo 的通用控制器之上。

    控制频率下，流程为：读取 XR -> 更新头/腰与双 TCP 目标 -> Placo 求 IK -> 写入
    MuJoCo position actuator -> 步进物理 -> 刷新主头部 Viewer 与两路腕部图像。
    """

    def __init__(
        self,
        xml_path: str,
        robot_urdf_path: str,
        scale_factor: float = 1.2,
        control_hz: float = 100.0,
        home_duration: float = 2.0,
        visualize_placo: bool = False,
        debug_xr: bool = False,
        tracker_serial: str = DEFAULT_TRACKER_SERIAL,
        show_wrist_viewers: bool = True,
        record: bool = False,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
        repo_id: str = DEFAULT_REPO_ID,
        task: str = DEFAULT_TASK,
        record_fps: int = 30,
    ) -> None:
        """加载模型、建立名称索引、初始化 IK 任务和所有安全状态。

        此处保存 ``home`` keyframe 作为 B 回零、腕臂姿态保持和头/腰中性姿态的唯一来源；
        因此 XML 中 keyframe 的 qpos/ctrl 修改会自动反映到遥操作初始状态。
        """
        if control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if home_duration <= 0.0:
            raise ValueError("home_duration must be positive")

        self.control_period = 1.0 / control_hz
        self.home_duration = home_duration
        self.debug_xr = debug_xr
        self.tracker_serial = tracker_serial
        self.show_wrist_viewers = show_wrist_viewers
        self.next_xr_debug_time = 0.0
        self.calibrated = {side: False for side in SIDES}
        self.grip_was_pressed = {side: False for side in SIDES}
        self.grip_release_required = {side: False for side in SIDES}
        self.button_was_pressed = {"X": False, "A": False, "B": False}
        self.gripper_command = {side: 1.0 for side in SIDES}  # 1 表示完全张开
        self.recorder = None  # --record 时在 __init__ 末尾创建
        self.homing = False
        self.homing_start_time = 0.0
        self.homing_start_ctrl = None
        self.home_ctrl = None
        self.headset_reference_rotation = None
        self.tracker_reference_rotation = None
        self.trunk_joint_targets = None
        self.tracker_available = False

        # 初版 H1 集成使用保守的笛卡尔工作空间、位移和转角安全限制。
        self.workspace_min = np.array([-0.05, -0.80, 0.40])
        self.workspace_max = np.array([1.00, 0.80, 1.50])
        self.max_translation_from_anchor = 0.50
        self.max_rotation_from_anchor = math.radians(150.0)
        self.max_linear_speed = 1.0
        self.max_angular_speed = math.radians(180.0)

        super().__init__(
            xml_path=xml_path,
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config(),
            floating_base=False,
            R_headset_world=R_HEADSET_TO_WORLD,
            visualize_placo=visualize_placo,
            scale_factor=scale_factor,
            dt=self.control_period,
        )

        home_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise ValueError("MuJoCo model does not contain the 'home' keyframe")
        self.home_keyframe_id = home_id
        self.home_ctrl = self.mj_model.key_ctrl[home_id].copy()

        self.head_camera_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME
        )
        if self.head_camera_id < 0:
            raise ValueError(f"Missing MuJoCo camera: {HEAD_CAMERA_NAME}")
        self.wrist_camera_names = dict(WRIST_CAMERA_NAMES)
        for camera_name in self.wrist_camera_names.values():
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                raise ValueError(f"Missing MuJoCo camera: {camera_name}")

        self.robot_actuator_ids = {}
        self.home_joint_positions = {}
        for joint_name in UPPER_BODY_JOINTS:
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_name = f"{joint_name}_position"
            actuator_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"Missing MuJoCo joint/actuator for '{joint_name}'")
            self.robot_actuator_ids[joint_name] = actuator_id
            qpos_address = self.mj_model.jnt_qposadr[joint_id]
            self.home_joint_positions[joint_name] = float(
                self.mj_model.key_qpos[home_id, qpos_address]
            )

        self.gripper_actuator_ids = {}
        for side, config in self.manipulator_config.items():
            actuator_id = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                config["gripper_actuator"],
            )
            if actuator_id < 0:
                raise ValueError(f"Missing gripper actuator: {config['gripper_actuator']}")
            self.gripper_actuator_ids[side] = actuator_id

        # 双 TCP 任务同时求解时，将未直接控制的上半身自由度保持在 home 附近；
        # TCP 任务保留通用控制器中配置的更高权重。
        self.posture_task = self.solver.add_joints_task()
        self.posture_task.set_joints(self.home_joint_positions)
        self.posture_task.configure("h1_home_posture", "soft", 1e-3)

        self.trunk_joint_targets = {
            name: self.home_joint_positions[name] for name in ("waist_J1", "waist_J2", "head_j1", "Head_j2")
        }
        self.trunk_task = self.solver.add_joints_task()
        self.trunk_task.set_joints(
            {name: self.home_joint_positions[name] for name in ("waist_J1", "waist_J2", "head_j1", "Head_j2")}
        )
        self.trunk_task.configure("h1_trunk_tracking", "hard", 1.0)

        # 未激活手臂在关节空间保持，因此会随腰部自然运动，而不会为了维持世界固定 TCP
        # 进行反向补偿。
        self.arm_hold_task = {}
        for side in SIDES:
            self.arm_hold_task[side] = self.solver.add_joints_task()
            self.arm_hold_task[side].set_joints(
                {joint: self.home_joint_positions[joint] for joint in ARM_JOINTS[side]}
            )
            self._configure_arm_tasks(side, tcp_active=False)

        self.sync_end_effector_poses_to_placo_tasks()
        for side, actuator_id in self.gripper_actuator_ids.items():
            self.mj_data.ctrl[actuator_id] = self.gripper_command[side]

        print("\nH1 PICO teleoperation ready")
        print("  Hold Grip: align TCP and move that arm")
        print("  Release Grip: freeze      Trigger: close gripper")
        print(f"  Head FIXED at home (yaw={self.home_joint_positions['head_j1']:.2f}, "
              f"pitch={self.home_joint_positions['Head_j2']:.2f}rad looking at table); "
              f"headset does not steer head; Tracker {self.tracker_serial} -> waist yaw/pitch")
        print("  A: reset both red cubes to their home poses")
        print("  B: smooth home + disarm both arms")
        print("  After B, release Grip once and press it again.\n")
        if self.show_wrist_viewers:
            print("  Viewers: main=head_rgb; two wrist RGB windows at 512x512 / 30 FPS\n")
        if self.debug_xr:
            print("XR debug output enabled (2 Hz). Move controllers and press buttons to inspect the stream.\n")

        # --record：惰性 import 创建记录器，未开录制时完全不依赖 lerobot。
        if record:
            from h1_dataset_recorder import H1DatasetRecorder

            self.recorder = H1DatasetRecorder(
                self.mj_model,
                repo_id=repo_id,
                root=dataset_dir,
                task=task,
                fps=record_fps,
            )
            print("  X: toggle recording (start episode / save episode)")
            print("  B while recording: save episode; A while recording: discard it\n")

    def _button_rising_edge(self, name: str) -> bool:
        """检测一次性按键上升沿，避免 A/B 被持续按住时每个控制周期重复触发。"""
        pressed = bool(self.xr_client.get_button_state_by_name(name))
        rising = pressed and not self.button_was_pressed[name]
        self.button_was_pressed[name] = pressed
        return rising

    def _print_xr_debug(self) -> None:
        """以 2 Hz 输出手柄、头部、Tracker 与上半身目标，便于检查 PICO 数据链路。"""
        if not self.debug_xr or self.mj_data.time < self.next_xr_debug_time:
            return
        self.next_xr_debug_time = float(self.mj_data.time) + 0.5
        left_pose = np.asarray(self.xr_client.get_pose_by_name("left_controller"), dtype=float)
        right_pose = np.asarray(self.xr_client.get_pose_by_name("right_controller"), dtype=float)
        timestamp = self.xr_client.get_timestamp_ns()
        print(
            "[XR] "
            f"ts={timestamp} X={int(self.button_was_pressed['X'])} "
            f"A={int(self.button_was_pressed['A'])} B={int(self.button_was_pressed['B'])} | "
            f"L p={np.round(left_pose[:3], 3)} grip="
            f"{self.xr_client.get_key_value_by_name('left_grip'):.2f} "
            f"trigger={self.xr_client.get_key_value_by_name('left_trigger'):.2f} | "
            f"R p={np.round(right_pose[:3], 3)} grip="
            f"{self.xr_client.get_key_value_by_name('right_grip'):.2f} "
            f"trigger={self.xr_client.get_key_value_by_name('right_trigger'):.2f}",
            flush=True,
        )
        print(
            "[BODY] "
            f"tracker={int(self.tracker_available)} "
            f"waist=({self.trunk_joint_targets['waist_J1']:.3f}, "
            f"{self.trunk_joint_targets['waist_J2']:.3f}) "
            f"head=({self.trunk_joint_targets['head_j1']:.3f}, "
            f"{self.trunk_joint_targets['Head_j2']:.3f})",
            flush=True,
        )

    def _configure_arm_tasks(self, side: str, tcp_active: bool) -> None:
        """在笛卡尔 TCP 跟踪与关节空间保持之间切换某一只手臂。

        Grip 按住时优先 TCP IK；松开后优先关节保持，防止重力导致末端下垂。
        """
        tcp_weight = 1.0 if tcp_active else 0.0
        hold_weight = 1e-4 if tcp_active else 1.0
        self.effector_task[side].configure(side, "soft", tcp_weight)
        self.arm_hold_task[side].configure(f"{side}_arm_hold", "soft", hold_weight)

    def _capture_arm_hold_target(self, side: str, use_home: bool = False) -> None:
        """把当前（或 home）关节角存为松开 Grip 后的保持目标。"""
        if use_home:
            targets = {joint: self.home_joint_positions[joint] for joint in ARM_JOINTS[side]}
        else:
            targets = {joint: float(self.placo_robot.get_joint(joint)) for joint in ARM_JOINTS[side]}
        self.arm_hold_task[side].set_joints(targets)

    def _xr_pose_world_rotation(self, pose: np.ndarray) -> np.ndarray | None:
        """将 XR 的 [xyz, xyzw] 姿态旋转转换到 MuJoCo 世界坐标，忽略无效输入。"""
        if not self._controller_pose_is_valid(pose):
            return None
        quaternion = np.array([pose[6], pose[3], pose[4], pose[5]], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        xr_rotation = tf.quaternion_matrix(quaternion)[:3, :3]
        return self.R_headset_world @ xr_rotation @ self.R_headset_world.T

    @staticmethod
    def _relative_yaw_pitch(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
        """计算 current 相对 reference 的 yaw/pitch，供头显和腰部 Tracker 使用。"""
        relative = np.eye(4)
        relative[:3, :3] = current @ reference.T
        _, pitch, yaw = tf.euler_from_matrix(relative, axes="sxyz")
        return float(yaw), float(pitch)

    def _rate_limit_joint_target(self, joint: str, desired: float, maximum_speed: float) -> float:
        """限制每个控制周期的关节目标变化，避免 XR 抖动直接传给机器人。"""
        current = self.trunk_joint_targets[joint]
        maximum_step = maximum_speed * self.control_period
        return float(current + np.clip(desired - current, -maximum_step, maximum_step))

    def _update_upper_body_targets(self) -> None:
        """把腰部 Tracker 相对旋转映射到 waist；v4 起头部完全固定。

        HEAD_FIXED_AT_HOME=True（默认）时头部永远保持 home keyframe 姿态
        (yaw=0、Head_j2=0.53 正对桌面)，头显旋转不再联动机器人头部 ——
        头部相机成为固定 overlooking 相机，回合间/回合内视角零散射。
        False 时恢复旧行为：头显相对旋转映射到头部 yaw/pitch（参考姿态在
        首次收到有效数据时锁定，用户的自然站姿就是零位）。
        waist_J2 是单向限位，故只允许 [0, 1.5708] rad，不会随 Tracker 后仰而越界。
        """
        headset_pose = np.asarray(self.xr_client.get_pose_by_name("headset"), dtype=float)
        headset_rotation = self._xr_pose_world_rotation(headset_pose)
        if headset_rotation is not None or HEAD_FIXED_AT_HOME:
            if HEAD_FIXED_AT_HOME:
                desired_head_yaw = self.home_joint_positions["head_j1"]
                desired_head_pitch = self.home_joint_positions["Head_j2"]
            else:
                if self.headset_reference_rotation is None:
                    self.headset_reference_rotation = headset_rotation.copy()
                    print("Headset reference captured; head tracking enabled")
                head_yaw, head_pitch = self._relative_yaw_pitch(self.headset_reference_rotation, headset_rotation)
                desired_head_yaw = float(np.clip(-head_yaw, -math.pi, math.pi))
                desired_head_pitch = float(np.clip(head_pitch, -0.7854, 0.7854))
            self.trunk_joint_targets["head_j1"] = self._rate_limit_joint_target(
                "head_j1", desired_head_yaw, maximum_speed=2.0
            )
            self.trunk_joint_targets["Head_j2"] = self._rate_limit_joint_target(
                "Head_j2", desired_head_pitch, maximum_speed=1.5
            )

        tracker_data = self.xr_client.get_motion_tracker_data()
        self.tracker_available = self.tracker_serial in tracker_data
        if self.tracker_available:
            tracker_pose = np.asarray(tracker_data[self.tracker_serial]["pose"], dtype=float)
            tracker_rotation = self._xr_pose_world_rotation(tracker_pose)
            if tracker_rotation is not None:
                if self.tracker_reference_rotation is None:
                    self.tracker_reference_rotation = tracker_rotation.copy()
                    print(f"Tracker reference captured: {self.tracker_serial}; waist tracking enabled")
                waist_yaw, waist_pitch = self._relative_yaw_pitch(self.tracker_reference_rotation, tracker_rotation)
                desired_waist_yaw = float(np.clip(-waist_yaw, -math.pi, math.pi))
                # 严格保留提供的 waist_J2 单侧限位，不擅自扩展到负方向。
                desired_waist_pitch = float(np.clip(waist_pitch, 0.0, 1.5708))
                self.trunk_joint_targets["waist_J1"] = self._rate_limit_joint_target(
                    "waist_J1", desired_waist_yaw, maximum_speed=1.2
                )
                self.trunk_joint_targets["waist_J2"] = self._rate_limit_joint_target(
                    "waist_J2", desired_waist_pitch, maximum_speed=0.8
                )

        self.trunk_task.set_joints(self.trunk_joint_targets)

    def _controller_pose_is_valid(self, pose: np.ndarray) -> bool:
        """过滤未跟踪/NaN 手柄数据；有效姿态必须为 7 元素且四元数模长正常。"""
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            return False
        quaternion_norm = np.linalg.norm(pose[3:7])
        return quaternion_norm > 0.5

    def _set_task_to_current_tcp(self, side: str) -> None:
        """将指定 TCP 的 IK 目标设为其当前测量姿态，用于无跳变切换任务。"""
        xyz, quat = self._get_link_pose(self.manipulator_config[side]["link_name"])
        target = tf.quaternion_matrix(quat)
        target[:3, 3] = xyz
        self.effector_task[side].T_world_frame = target

    def _anchor_controller_to_tcp(self, side: str, announce: bool = False) -> bool:
        """在 Grip 刚按下时建立手柄相对零点与当前 TCP 的对应关系。

        这不是绝对世界坐标标定：手柄当前位置不会拉动机械臂；之后只有手柄的相对变化才
        转化为 TCP 增量，因此可随时松开再抓握（clutch）。
        """
        config = self.manipulator_config[side]
        xr_pose = np.asarray(self.xr_client.get_pose_by_name(config["pose_source"]), dtype=float)
        if not self._controller_pose_is_valid(xr_pose):
            print(f"{side}: ignored calibration because controller pose is invalid")
            return False

        self.ref_controller_xyz[side] = None
        self.ref_controller_quat[side] = None
        self._process_xr_pose(xr_pose, side)
        self.ref_ee_xyz[side], self.ref_ee_quat[side] = self._get_link_pose(config["link_name"])
        self._set_task_to_current_tcp(side)

        if announce:
            self.calibrated[side] = True
            print(f"{side}: controller aligned to {config['link_name']}; hold Grip to move")
        return True

    @staticmethod
    def _clip_vector_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        """保留方向地裁剪向量长度。"""
        magnitude = np.linalg.norm(vector)
        if magnitude <= maximum or magnitude < 1e-12:
            return vector
        return vector * (maximum / magnitude)

    def _limit_target_pose(self, side: str, xyz: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对目标 TCP 同时施加工作空间、线速度和角速度三重安全限制。"""
        xyz = np.clip(xyz, self.workspace_min, self.workspace_max)
        current_target = self.effector_task[side].T_world_frame
        current_xyz = current_target[:3, 3]
        current_quat = tf.quaternion_from_matrix(current_target)

        step = xyz - current_xyz
        xyz = current_xyz + self._clip_vector_norm(step, self.max_linear_speed * self.control_period)

        angular_step = np.linalg.norm(quat_diff_as_angle_axis(current_quat, quat))
        max_angular_step = self.max_angular_speed * self.control_period
        if angular_step > max_angular_step:
            quat = tf.quaternion_slerp(current_quat, quat, max_angular_step / angular_step)
        return xyz, np.asarray(quat)

    def _start_homing(self) -> None:
        """响应 B 键：解除双臂控制，清除 XR 参考，并开始平滑回到 home。"""
        if self.recorder is not None and self.recorder.recording:
            self.recorder.save_episode()  # 回零过渡不属于演示数据
        self.calibrated = {side: False for side in SIDES}
        self.active = {side: False for side in SIDES}
        self.grip_was_pressed = {side: False for side in SIDES}
        self.grip_release_required = {side: True for side in SIDES}
        self.headset_reference_rotation = None
        self.tracker_reference_rotation = None
        self.tracker_available = False
        self.trunk_joint_targets = {
            name: self.home_joint_positions[name] for name in ("waist_J1", "waist_J2", "head_j1", "Head_j2")
        }
        self.trunk_task.set_joints(self.trunk_joint_targets)
        for side in SIDES:
            self._configure_arm_tasks(side, tcp_active=False)
        self.homing = True
        self.homing_start_time = float(self.mj_data.time)
        self.homing_start_ctrl = self.mj_data.ctrl.copy()
        print("B: both arms disarmed; returning the upper body to home")

    def _reset_red_cubes(self) -> None:
        """响应 A 键：把两个 freejoint 红方块的位姿、速度和外力恢复为 home keyframe。

        v4 固定出生点 = (0.55, ±0.12)(双臂之间,头部下俯视野正中),直接定义在
        H1_scene.xml 的 home keyframe 里 —— rollout --spawn home 与本函数自动同源。
        (曾试过随机出生带,遥操作实测部分落点不好抓,已改回固定;随机带基础设施
        保留在 teleop/h1_spawn_zones.py 备用。)
        """
        for cube_name in ("red_cube_left", "red_cube_right"):
            joint_name = f"{cube_name}_freejoint"
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
            if joint_id < 0 or body_id < 0:
                print(f"A: cannot reset missing free body '{cube_name}'")
                continue

            qpos_address = self.mj_model.jnt_qposadr[joint_id]
            dof_address = self.mj_model.jnt_dofadr[joint_id]
            self.mj_data.qpos[qpos_address : qpos_address + 7] = self.mj_model.key_qpos[
                self.home_keyframe_id, qpos_address : qpos_address + 7
            ]
            self.mj_data.qvel[dof_address : dof_address + 6] = 0.0
            self.mj_data.qacc[dof_address : dof_address + 6] = 0.0
            self.mj_data.qacc_warmstart[dof_address : dof_address + 6] = 0.0
            self.mj_data.xfrc_applied[body_id] = 0.0

        mujoco.mj_forward(self.mj_model, self.mj_data)
        print("A: both red cubes reset to home keyframe poses (固定点 0.65, ±0.30)")

    def _finish_homing(self) -> None:
        """回零完成后重置控制锚点；必须先松开再按 Grip，才能重新接管机械臂。"""
        self.homing = False
        self._update_robot_state()
        for side in SIDES:
            self.ref_ee_xyz[side] = None
            self.ref_ee_quat[side] = None
            self.ref_controller_xyz[side] = None
            self.ref_controller_quat[side] = None
            self._set_task_to_current_tcp(side)
            self._capture_arm_hold_target(side, use_home=True)
            self._configure_arm_tasks(side, tcp_active=False)
        print("Home reached. Release Grip, then press it again to control each arm.")

    def _update_ik(self) -> None:
        """一个控制周期的上层逻辑：按键、回零、上半身、双臂 clutch 与 Placo 求解。"""
        self._update_robot_state()
        self.placo_robot.update_kinematics()

        # X 切换录制；A 重置两个 freejoint 红色方块；B 回零。
        x_edge = self._button_rising_edge("X")
        a_edge = self._button_rising_edge("A")
        b_edge = self._button_rising_edge("B")
        self._print_xr_debug()

        if x_edge:
            if self.recorder is None:
                print("X: recording disabled; restart with --record to enable it")
            else:
                self.recorder.toggle()

        if a_edge:
            if self.recorder is not None and self.recorder.recording:
                # 方块瞬移会污染当前 episode，先丢弃再重置。
                self.recorder.discard_episode()
            self._reset_red_cubes()

        if b_edge:
            self._start_homing()

        if self.homing:
            elapsed = float(self.mj_data.time) - self.homing_start_time
            if elapsed >= self.home_duration:
                max_error = 0.0
                for joint_name in UPPER_BODY_JOINTS:
                    joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    qpos_address = self.mj_model.jnt_qposadr[joint_id]
                    error = abs(self.mj_data.qpos[qpos_address] - self.home_joint_positions[joint_name])
                    max_error = max(max_error, error)
                if max_error < 0.03 or elapsed >= self.home_duration + 2.0:
                    self._finish_homing()
            return

        self._update_upper_body_targets()

        for side, config in self.manipulator_config.items():
            grip_pressed = self.xr_client.get_key_value_by_name(config["control_trigger"]) > 0.8

            if self.grip_release_required[side]:
                self.active[side] = False
                if not grip_pressed:
                    self.grip_release_required[side] = False
                self.grip_was_pressed[side] = grip_pressed
                continue

            if grip_pressed:
                # 每次 Grip 上升沿都以当前 TCP 建立手柄相对零点，符合 XRoboToolkit 的
                # 离合/使能交互：手柄绝对位置不会造成机械臂跳变。
                if not self.grip_was_pressed[side]:
                    self.calibrated[side] = self._anchor_controller_to_tcp(side, announce=True)
                    if self.calibrated[side]:
                        self._configure_arm_tasks(side, tcp_active=True)

                self.active[side] = self.calibrated[side]

                xr_pose = np.asarray(self.xr_client.get_pose_by_name(config["pose_source"]), dtype=float)
                if self.active[side] and self._controller_pose_is_valid(xr_pose):
                    delta_xyz, delta_rot = self._process_xr_pose(xr_pose, side)
                    delta_xyz = self._clip_vector_norm(delta_xyz, self.max_translation_from_anchor)
                    delta_rot = self._clip_vector_norm(delta_rot, self.max_rotation_from_anchor)
                    target_xyz, target_quat = apply_delta_pose(
                        np.asarray(self.ref_ee_xyz[side]),
                        np.asarray(self.ref_ee_quat[side]),
                        delta_xyz,
                        delta_rot,
                    )
                    target_xyz, target_quat = self._limit_target_pose(side, target_xyz, target_quat)
                    target = tf.quaternion_matrix(target_quat)
                    target[:3, 3] = target_xyz
                    self.effector_task[side].T_world_frame = target
            else:
                self.active[side] = False
                if self.grip_was_pressed[side]:
                    # 保持最后一个 IK 任务和关节命令目标；不能追随受重力影响的测量姿态，
                    # 否则机械臂会逐渐下垂。
                    self.calibrated[side] = False
                    self._capture_arm_hold_target(side)
                    self._configure_arm_tasks(side, tcp_active=False)
                    self.ref_controller_xyz[side] = None
                    self.ref_controller_quat[side] = None
                    print(f"{side}: Grip released; holding the last target")

            self.grip_was_pressed[side] = grip_pressed

        try:
            self.solver.solve(True)
        except RuntimeError as error:
            print(f"IK solver failed; holding the previous command: {error}")

    def _update_gripper_target(self) -> None:
        """将左右 Trigger 映射成夹爪命令：0=open、1=closed，并消除很小的手柄噪声。"""
        if self.homing:
            return
        for side, config in self.manipulator_config.items():
            trigger = float(self.xr_client.get_key_value_by_name(config["gripper_trigger"]))
            if not math.isfinite(trigger):
                continue
            trigger = float(np.clip(trigger, 0.0, 1.0))
            if trigger < 0.03:
                trigger = 0.0
            elif trigger > 0.97:
                trigger = 1.0
            # 斜坡限速（复用头/腰的 _rate_limit_joint_target 思路）：让指令
            # 从上一周期的值渐变过去，记录的 action 不再是瞬跳阶跃。
            desired = 1.0 - trigger
            step = GRIPPER_COMMAND_SPEED * self.control_period
            current = self.gripper_command[side]
            self.gripper_command[side] = float(
                np.clip(desired, current - step, current + step)
            )

    def _send_command(self) -> None:
        """写入 MuJoCo actuator ctrl；回零期间对上半身采用三次平滑插值。"""
        if self.homing:
            elapsed = max(0.0, float(self.mj_data.time) - self.homing_start_time)
            phase = min(1.0, elapsed / self.home_duration)
            smooth_phase = phase * phase * (3.0 - 2.0 * phase)
            for joint_name, actuator_id in self.robot_actuator_ids.items():
                start = self.homing_start_ctrl[actuator_id]
                goal = self.home_ctrl[actuator_id]
                self.mj_data.ctrl[actuator_id] = start + smooth_phase * (goal - start)
        else:
            super()._send_command()

        # 回零时保持当前夹爪命令；按 B 不会主动松开已抓住的物体。
        for side, actuator_id in self.gripper_actuator_ids.items():
            self.mj_data.ctrl[actuator_id] = self.gripper_command[side]

    def run(self) -> None:
        """启动实时仿真与图形界面，直到主 MuJoCo 窗口关闭或 Ctrl-C。

        主 Viewer 固定到 head_rgb；两个 OpenCV 腕部窗口由同一个 mj_data 在 30 FPS
        渲染，避免视觉观测与机器人状态出现跨帧不同步。
        """
        simulation_timestep = float(self.mj_model.opt.timestep)
        next_control_time = float(self.mj_data.time)
        next_viewer_sync_time = float(self.mj_data.time)
        next_wrist_render_time = float(self.mj_data.time)
        next_record_time = float(self.mj_data.time)
        viewer_period = 1.0 / 60.0
        wrist_render_period = 1.0 / WRIST_CAMERA_FPS
        record_period = 1.0 / self.recorder.fps if self.recorder is not None else 0.0
        wall_start = time.monotonic()
        simulation_start = float(self.mj_data.time)
        wrist_renderer = None

        try:
            with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                # 主 MuJoCo 窗口固定使用机器人的头部相机。
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = self.head_camera_id

                if self.show_wrist_viewers:
                    wrist_renderer = mujoco.Renderer(
                        self.mj_model,
                        height=WRIST_CAMERA_HEIGHT,
                        width=WRIST_CAMERA_WIDTH,
                    )
                    for side, title in WRIST_VIEWER_TITLES.items():
                        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

                while viewer.is_running() and not self._stop_event.is_set():
                    if self.mj_data.time + 1e-12 >= next_control_time:
                        self._update_ik()
                        self._update_gripper_target()
                        self._update_mocap_target()
                        self._send_command()
                        next_control_time += self.control_period

                    mujoco.mj_step(self.mj_model, self.mj_data)
                    if self.mj_data.time + 1e-12 >= next_viewer_sync_time:
                        viewer.sync()
                        next_viewer_sync_time += viewer_period

                    if (
                        wrist_renderer is not None
                        and self.mj_data.time + 1e-12 >= next_wrist_render_time
                    ):
                        for side, camera_name in self.wrist_camera_names.items():
                            wrist_renderer.update_scene(self.mj_data, camera=camera_name)
                            image_rgb = wrist_renderer.render()
                            cv2.imshow(
                                WRIST_VIEWER_TITLES[side],
                                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                            )
                        # 处理 OpenCV 窗口事件，不会消耗 PICO 手柄输入。
                        cv2.waitKey(1)
                        next_wrist_render_time += wrist_render_period

                    if (
                        self.recorder is not None
                        and self.mj_data.time + 1e-12 >= next_record_time
                    ):
                        # 与腕部渲染同一模式的仿真时刻调度；未在录制时是空操作。
                        self.recorder.record_frame(self.mj_data)
                        next_record_time += record_period

                    target_wall_time = wall_start + (float(self.mj_data.time) - simulation_start)
                    remaining = target_wall_time - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(min(remaining, simulation_timestep))
        except KeyboardInterrupt:
            print("\nTeleoperation stopped.")
        finally:
            if self.recorder is not None:
                self.recorder.stop()  # 等后台编码完成, 保存未完回合并 finalize
            if wrist_renderer is not None:
                wrist_renderer.close()
            if self.show_wrist_viewers:
                for title in WRIST_VIEWER_TITLES.values():
                    try:
                        cv2.destroyWindow(title)
                    except cv2.error:
                        # 用户可能已手动关闭某个腕部窗口。
                        pass
            self._stop_event.set()
            self.xr_client.close()
            print("XRoboToolkit SDK closed.")


def check_models(mjcf_path: Path, urdf_path: Path) -> int:
    """离线一致性检查：验证 home 状态下两种模型的左右 TCP 位姿完全相同。"""
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise ValueError("Missing MuJoCo home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    robot = placo.RobotWrapper(str(urdf_path))
    robot.state.q = calc_placo_q_from_mujoco_qpos(model, robot, data.qpos, floating_base=False)
    robot.update_kinematics()

    required_frames = ("left_omnipicker_tcp_link", "right_omnipicker_tcp_link")
    for name in required_frames:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0 or not robot.model.existFrame(name):
            raise ValueError(f"TCP frame is not shared by MJCF and URDF: {name}")
        mujoco_position = data.xpos[body_id]
        mujoco_rotation = data.xmat[body_id].reshape(3, 3)
        placo_pose = robot.get_T_world_frame(name)
        position_error = np.linalg.norm(mujoco_position - placo_pose[:3, 3])
        rotation_error = np.linalg.norm(mujoco_rotation - placo_pose[:3, :3])
        if position_error > 1e-8 or rotation_error > 1e-8:
            raise ValueError(
                f"MJCF/URDF TCP mismatch for {name}: position={position_error:.3g}, rotation={rotation_error:.3g}"
            )
    print(f"Model check passed: {model.nu} MuJoCo actuators, {robot.model.nq - 7} Placo joints")
    print("Shared TCP frames: " + ", ".join(required_frames))
    return 0


def parse_args() -> argparse.Namespace:
    """定义运行、调试与关闭腕部窗口所需的命令行选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--scale-factor", type=float, default=1.2)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--home-duration", type=float, default=2.0)
    parser.add_argument("--visualize-placo", action="store_true")
    parser.add_argument(
        "--tracker-serial",
        default=DEFAULT_TRACKER_SERIAL,
        help="serial number of the waist Motion Tracker",
    )
    parser.add_argument(
        "--debug-xr",
        action="store_true",
        help="print controller poses and button values at 2 Hz from this same SDK connection",
    )
    parser.add_argument(
        "--no-wrist-viewers",
        action="store_true",
        help="do not open the two live wrist-camera windows",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="enable X-button recording into a LeRobotDataset (X: start/save episode)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="dataset root for --record; an existing dataset resumes, a new dir creates one",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="dataset repo id, e.g. mjq/h1_sim_pick_bin",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="task string stored with every recorded frame",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=30,
        help="recording frame rate scheduled in simulation time",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="validate MJCF/URDF names without starting XR or the viewer",
    )
    return parser.parse_args()


def main() -> int:
    """校验输入路径并创建遥操作控制器。"""
    args = parse_args()
    if not args.mjcf.is_file():
        print(f"MJCF does not exist: {args.mjcf}", file=sys.stderr)
        return 2
    if not args.urdf.is_file():
        print(f"URDF does not exist: {args.urdf}", file=sys.stderr)
        return 2
    if args.check_model:
        return check_models(args.mjcf.resolve(), args.urdf.resolve())

    controller = H1PicoTeleopController(
        xml_path=str(args.mjcf.resolve()),
        robot_urdf_path=str(args.urdf.resolve()),
        scale_factor=args.scale_factor,
        control_hz=args.control_hz,
        home_duration=args.home_duration,
        visualize_placo=args.visualize_placo,
        debug_xr=args.debug_xr,
        tracker_serial=args.tracker_serial,
        show_wrist_viewers=not args.no_wrist_viewers,
        record=args.record,
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        task=args.task,
        record_fps=args.record_fps,
    )
    controller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ==================== 数采命令速查（2026-08-20，头部俯视 0.4 姿态版） ====================
#
# 前置（PICO 链路）：
#   1) /opt/apps/roboticsservice/runService.sh     # 启动 PC Service
#   2) PICO 戴上并亮屏（摘头显=断连=数据全 0），XRoboToolkit 连电脑 IP，Controller Tracking ON
#   3) ss -tn | grep 63901                         # 确认数据流已连接
#
# 纯遥操（不录数据）：
#   conda activate xr-robotics
#   cd ~/robot_item/lerobot-main/H1_simulation_model/H1_build
#   python teleop/h1_pico_teleop.py
#
# 数采（v2 新数据集；目录不要提前创建，存在会转续采）：
#   conda activate xr-robotics
#   cd ~/robot_item/lerobot-main/H1_simulation_model/H1_build
#   python teleop/h1_pico_teleop.py --record \
#       --repo-id mjq/h1_build_pick_v2 \
#       --dataset-dir data/h1_build_pick_v2
#
# 快捷启动器（在 H1_build 目录下直接运行，等价于上面两条命令）：
#   ./run_teleop.py     # 纯遥操
#   ./run_record.py     # 数采（v2 数据集）
#
# 每回合节奏：A 重置方块 → 等 1s 落稳 → X 开始录制 → 完成抓放入桶 → X 保存
#   * 录制中按 A = 丢弃当前回合（防方块瞬移污染数据）
#   * B = 双臂回 home（录制中按 B 自动保存）
#   * 换任务语句：--task "..."；帧率：--record-fps（默认 30）
#   * 手柄没反应：--debug-xr 看原始数据
# ====================================================================
