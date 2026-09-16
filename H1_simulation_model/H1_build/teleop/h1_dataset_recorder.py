#!/usr/bin/env python3
"""H1 MuJoCo 遥操作数据采集记录器 —— 把仿真状态与三路相机写进 LeRobotDataset。

文件关系
--------
* 本模块只依赖 mujoco / numpy / lerobot，不 import XRoboToolkit，
  因此既能在 PICO 遥操作（``h1_pico_teleop.py``）里用，
  也能在无头验证脚本（``h1_record_check.py``）里独立跑通。
* 数据内容（对应 README 第 11 节的建议，取其子集）：
    - ``observation.images.head_rgb / left_wrist_rgb / right_wrist_rgb``
      512x512 RGB，按 episode 编码为 mp4（lerobot v3.0 视频格式）；
    - ``observation.state``：上半身 18 关节 qpos + 左右夹爪开合测量值；
    - ``action``：对应的 18 路关节位置 ctrl + 2 路夹爪开合 ctrl。
  图像、state、action 在同一个 MuJoCo 仿真时刻采样，天然时间对齐。

lerobot 0.5.2 API 要点（踩过的坑写在这里，改代码前先读）
--------
* ``LeRobotDataset.create`` 要求 root 目录不存在；已存在时用 ``resume`` 续采。
* 帧字典的 key 必须与 features 一一对应，外加每帧一个 ``task`` 字符串；
  禁止传 ``timestamp`` / ``frame_index`` 等保留键（自动生成）。
* 数组必须严格 ``float32``（float64 会直接被拒）。
* 全部采完后必须 ``finalize()``，否则数据集读不回来。
* ``add_frame`` 时图像立即落成临时 PNG（磁盘缓冲，不占内存），
  ``save_episode`` 才编码 mp4 并清理临时文件 —— 所以长回合不会撑爆内存。

用法（在遥操作主循环里）
--------
    recorder = H1DatasetRecorder(model, repo_id="mjq/h1_sim_pick_bin", root="...")
    recorder.start_episode()            # X 键第一次按下
    ... 每 1/fps 秒调用 recorder.record_frame(data) ...
    recorder.save_episode()             # X 键再按一次 / B 回零时
    recorder.stop()                     # 程序退出前（内部会 finalize）
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 三台相机名称必须与 MJCF 中 <camera name="..."> 完全一致。
CAMERA_NAMES = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
IMAGE_SIZE = 512
DEFAULT_REPO_ID = "mjq/h1_build_pick"
DEFAULT_TASK = "Pick up the red cube and put it into the green bin"
# state/action 的关节顺序；Right_j6 的小写 j 是模型历史命名，勿"修正"。
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
GRIPPER_ACTUATOR_NAMES = {
    "left": "left_omnipicker_gripper_opening",
    "right": "right_omnipicker_gripper_opening",
}


def make_features() -> dict:
    """返回 LeRobotDataset 的 features 定义（video 图像 + 20 维 state/action）。"""
    names = list(UPPER_BODY_JOINTS) + ["left_gripper.open", "right_gripper.open"]
    features = {
        "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        "action": {"dtype": "float32", "shape": [len(names)], "names": names},
    }
    for camera in CAMERA_NAMES:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
            "names": ["channels", "height", "width"],
        }
    return features


class H1DatasetRecorder:
    """把 H1 仿真状态 + 三相机图像记录为 LeRobotDataset 的独立记录器。

    线程模型：``record_frame`` 由仿真主线程按 fps 调用；``save_episode``
    的视频编码在后台线程串行执行（编码期间主循环照常跑物理），
    ``_lock`` 保证 add_frame 与 save/clear 不会交错。GL 上下文惰性创建，
    避免在 mujoco.viewer 启动之前抢占上下文。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        repo_id: str = DEFAULT_REPO_ID,
        root: str | Path = "",
        task: str = DEFAULT_TASK,
        fps: int = 30,
        image_size: int = IMAGE_SIZE,
        robot_type: str = "h1_omnipicker",
        vcodec: str = "libsvtav1",
        image_writer_threads: int = 4,
    ) -> None:
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"fps 必须是正整数, 得到 {fps!r}")
        if image_size > model.vis.global_.offwidth or image_size > model.vis.global_.offheight:
            raise ValueError(
                f"图像尺寸 {image_size} 超出模型离屏缓冲 "
                f"({model.vis.global_.offwidth}x{model.vis.global_.offheight})"
            )

        self.model = model
        self.repo_id = repo_id
        self.root = Path(root) if root else Path("data") / repo_id.split("/")[-1]
        self.task = task
        self.fps = fps
        self.image_size = image_size

        # 名称索引：关节 qpos 地址、关节执行器、夹爪执行器、相机 id。
        self._joint_qposadr = []
        self._joint_actuators = []
        for joint_name in UPPER_BODY_JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_position")
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"模型缺少关节/执行器: '{joint_name}'")
            self._joint_qposadr.append(int(model.jnt_qposadr[joint_id]))
            self._joint_actuators.append(actuator_id)
        self._gripper_actuators = []
        for side, actuator_name in GRIPPER_ACTUATOR_NAMES.items():
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                raise ValueError(f"模型缺少夹爪执行器: '{actuator_name}'")
            self._gripper_actuators.append(actuator_id)
        self._camera_ids = {}
        for camera_name in CAMERA_NAMES:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"模型缺少相机: '{camera_name}'")
            self._camera_ids[camera_name] = camera_id

        # 数据集：目录已存在且至少保存过一个 episode 才续采，否则新建。
        # create() 之后一次没存就退出会留下只有 info.json 的"空壳"，
        # resume 读不到 meta/tasks.parquet 会转去 Hub 拉取并报 401 ——
        # 因此空壳直接删掉重建（tasks.parquet 只在首次 save_episode 时写出，
        # 它存在与否就是"是否可续采"的可靠判据）。
        resumable = self.root.exists() and (self.root / "meta" / "tasks.parquet").exists()
        if self.root.exists() and not resumable:
            print(f"[recorder] 检测到未保存过 episode 的空数据集, 删除重建: {self.root}")
            shutil.rmtree(self.root)
        if resumable:
            print(f"[recorder] 数据集已存在, 续采: {self.root}")
            self.dataset = LeRobotDataset.resume(
                repo_id=repo_id, root=str(self.root), vcodec=vcodec
            )
        else:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=str(self.root),
                features=make_features(),
                robot_type=robot_type,
                use_videos=True,
                vcodec=vcodec,
                image_writer_threads=image_writer_threads,
            )
            print(f"[recorder] 新建数据集: {self.root} (fps={fps})")

        self.recording = False
        self.frame_count = 0
        self.saved_episodes = 0
        self._renderers: dict[str, mujoco.Renderer] | None = None
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()  # 同一时刻只允许一个 save 在编码

    # ── episode 管理 ─────────────────────────────────────────

    def start_episode(self) -> None:
        """开始一个新回合；若上一回合还在编码会先等它写完（保证缓冲不串）。"""
        with self._encode_lock:
            with self._lock:
                if self.recording:
                    return
                self.recording = True
                self.frame_count = 0
                print(f"[recorder] 开始录制 episode (已保存 {self.saved_episodes} 回合)")

    def toggle(self) -> bool:
        """X 键语义：未录则开始，录制中则保存并结束。返回最新的录制状态。"""
        if self.recording:
            self.save_episode()
            return False
        self.start_episode()
        return True

    def save_episode(self) -> None:
        """保存当前回合（后台线程编码 mp4 + 写 parquet），立即返回。"""
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            frames = self.frame_count
            self.frame_count = 0
        if frames == 0:
            print("[recorder] 本回合 0 帧, 跳过保存")
            self._clear_buffer_locked()
            return

        def encode() -> None:
            with self._encode_lock:
                with self._lock:
                    if self.dataset.writer.episode_buffer["size"] == 0:
                        return
                    self.dataset.save_episode()
                    self.saved_episodes += 1
                    print(f"[recorder] episode 已保存 ({frames} 帧, 累计 {self.saved_episodes})")

        threading.Thread(target=encode, name="h1-episode-encoder", daemon=True).start()

    def discard_episode(self) -> None:
        """丢弃当前回合（A 键重置方块等事件会污染数据时使用）。"""
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            frames = self.frame_count
            self.frame_count = 0
            self._clear_buffer_locked()
            print(f"[recorder] 已丢弃当前 episode ({frames} 帧)")

    def _clear_buffer_locked(self) -> None:
        """清空 writer 的 episode 缓冲并删除该回合的临时图像目录。

        DatasetWriter.clear_episode_buffer 只清理 dtype=image 的临时图，
        video 特征的 PNG 目录需要按 video_keys 手动删。删目录前必须先等
        异步图像写队列排干 —— 否则残留写请求会在目录重建后落进新回合的
        同名 PNG（丢弃后立刻开录时会污染数据）。
        """
        writer = self.dataset.writer
        if writer.image_writer is not None:
            writer.image_writer.wait_until_done()
        episode_index = writer.episode_buffer["episode_index"]
        if isinstance(episode_index, np.ndarray):
            episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
        for video_key in writer._meta.video_keys:
            image_dir = writer._get_image_file_dir(episode_index, video_key)
            if image_dir.is_dir():
                shutil.rmtree(image_dir)
        writer.clear_episode_buffer(delete_images=False)

    def wait(self) -> None:
        """阻塞直到后台 episode 编码全部完成（stop 前内部也会调用）。"""
        with self._encode_lock:
            pass

    def stop(self) -> None:
        """退出前调用：保存未完回合、等编码队列传彻底、finalize 数据集。"""
        self.wait()
        if self.recording and self.frame_count > 0:
            # 退出路径没有下一回合, 直接同步保存。
            self.recording = False
            self.dataset.save_episode()
            self.saved_episodes += 1
        elif self.recording:
            self._clear_buffer_locked()
            self.recording = False
        self.dataset.finalize()
        if self._renderers is not None:
            for renderer in self._renderers.values():
                renderer.close()
            self._renderers = None
        print(f"[recorder] 已完成并 finalize, 共保存 {self.saved_episodes} 回合, 数据在 {self.root}")

    # ── 采帧 ─────────────────────────────────────────────────

    def record_frame(self, data: mujoco.MjData) -> bool:
        """渲染三相机 + 采 state/action, 写入当前 episode。

        由调用方按 1/fps 的仿真时刻调度（与 h1_pico_teleop.run 里
        腕部相机的 next_wrist_render_time 调度同一模式）。
        未在录制时直接返回 False, 不做任何渲染。
        """
        if not self.recording:
            return False
        if self._renderers is None:  # 惰性创建 GL 上下文（须在 viewer 启动后）
            self._renderers = {
                name: mujoco.Renderer(self.model, self.image_size, self.image_size)
                for name in CAMERA_NAMES
            }

        images = {}
        for name, renderer in self._renderers.items():
            renderer.update_scene(data, camera=self._camera_ids[name])
            images[name] = renderer.render()  # uint8 (H, W, 3) RGB

        state = np.empty(20, dtype=np.float32)
        for i, address in enumerate(self._joint_qposadr):
            state[i] = data.qpos[address]
        for i, actuator_id in enumerate(self._gripper_actuators):
            # position 执行器的 actuator_length 已被 gear 归一化到 0..1 开合度
            state[18 + i] = data.actuator_length[actuator_id]

        action = np.empty(20, dtype=np.float32)
        for i, actuator_id in enumerate(self._joint_actuators):
            action[i] = data.ctrl[actuator_id]
        for i, actuator_id in enumerate(self._gripper_actuators):
            action[18 + i] = data.ctrl[actuator_id]

        frame = {
            "observation.state": state,
            "action": action,
            "task": self.task,
        }
        for name, image in images.items():
            frame[f"observation.images.{name}"] = image

        with self._lock:
            if not self.recording:  # save/discard 恰好插进来了
                return False
            self.dataset.add_frame(frame)
            self.frame_count += 1
            return True
