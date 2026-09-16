#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''SmolVLA H1 仿真闭环推理（rollout）程序'''

"""
h1_smolvla_inference_rollout.py
用训练好的 SmolVLA 权重闭环控制 MuJoCo 里的 H1 双臂，完成
"Pick up the red cube and put it into the green bin"，并统计成功率。
参照 mjq_smolVLA/Dual_Arm_Smolvla_inference.py（SO101 双臂手写推理循环）改写。

直接运行（不加参数）= 遥操作数采同款观感：
  MuJoCo 主窗口（固定头部相机第一人称）+ 左右腕两个 512 小窗
  （"H1 left wrist RGB" / "H1 right wrist RGB"），30Hz 实时节奏，
  腕部窗口按 q 可提前结束当前回合，Ctrl+C 随时退出。
加 --headless = 无窗口全速批量评估成功率。

与训练/数采的三条对齐原则（改任何一条都会让策略表现崩坏）：
1. 观测组装与 h1_dataset_recorder.record_frame 逐位一致：
   - 三相机图像 + state 在同一个 MuJoCo 仿真时刻采样；
   - state 20 维 = 上身 18 关节 qpos（UPPER_BODY_JOINTS 顺序）
     + 左右夹爪 actuator_length（0..1 开合度），不是 qpos；
   - 图像 uint8 HWC → prepare_observation_for_inference 自动转 float/255 CHW。
2. 观测键名用训练时 rename_map 的目标名（camera1=左腕, camera2=右腕,
   camera3=头）——训练时数据集键已改名喂给策略，推理必须同名。
3. 动作执行与采集时 action 定义一致：20 维 = 18 路关节位置 ctrl
   + 2 路夹爪开合 ctrl，直接写 data.ctrl 对应执行器。

动作 chunk 语义：smolvla select_action 内部维护队列，一次推理产出
chunk_size=50 步，此后每帧弹一个、队列空了才重新推理（30Hz 下约
1.67s 推理一次），因此主循环每帧只管调用 select_action 拿单步动作。

回合流程（复刻遥操作数采的 B→A→X 节奏）：
  mj_resetDataKeyframe(home) → 保持 home ctrl 静置 0.6s（方块从 keyframe
  高度落到桌面并稳定，与数采时按 A 重置后的状态一致）→ policy.reset()
  → 30Hz 闭环 → 超过 max_steps 或双块入桶即结束。

成功判定：green_bin 位于 (0.7, 0)，内腔半宽 0.045m、底面 z≈0.646；
红方块边长 0.04m。方块中心满足 |x-0.7|<0.03 且 |y|<0.03 且
0.646<z<0.75 视为入桶。回合成功 = 结束时两块都在桶里
（另有 ever 指标 = 全程任一时刻入过桶，用于区分"放进又弹出"）。

用法（lerobot312 环境）：
    conda activate lerobot312
    cd ~/robot_item/lerobot-main
    # 默认：仿真窗口 + 左右腕小窗实时观看（30Hz 实时，1 回合）
    python H1_simulation_model/H1_build/train/h1_smolvla_inference_rollout.py
    # 无窗口全速批量统计成功率（10 回合）
    python H1_simulation_model/H1_build/train/h1_smolvla_inference_rollout.py --headless --episodes 10
    # 常用参数
    #   --episodes 5        连续回合数
    #   --max-steps 600     每回合最大帧数（数据集平均回合约 670 帧）
    #   --task "..."        换任务指令
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── 参数解析先于 import mujoco：无头模式强制 EGL 离屏渲染 ──────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA H1 MuJoCo closed-loop rollout")
    parser.add_argument("--checkpoint", type=str, default="", help="策略目录（默认 output_smolvla_h1_v3/checkpoints/last/pretrained_model）")
    parser.add_argument("--headless", action="store_true", help="无窗口全速批量评估（默认开仿真窗口+腕部小窗实时观看）")
    parser.add_argument("--episodes", type=int, default=0, help="评估回合数（默认：观看 1 回合 / 无头 10 回合）")
    parser.add_argument("--max-steps", type=int, default=0, help="每回合最大帧数（默认：观看 1500=50s / 无头 900=30s；数据集平均回合约 670 帧）")
    parser.add_argument("--task", type=str, default="", help="任务指令（默认取数采记录器里的 DEFAULT_TASK）")
    parser.add_argument("--device", type=str, default="cuda", help="cpu / cuda")
    parser.add_argument("--settle", type=float, default=0.6, help="回合开始前的静置秒数（等方块落稳）")
    args = parser.parse_args()
    if args.episodes <= 0:
        args.episodes = 10 if args.headless else 1
    if args.max_steps <= 0:
        args.max_steps = 900 if args.headless else 1500
    return args


ARGS = parse_args()
if ARGS.headless:
    os.environ.setdefault("MUJOCO_GL", "egl")  # 无头离屏渲染（本机已验证 egl 可用）

# cv2 必须先于 mujoco 导入（遥操作同款顺序）：viewer/GL 上下文建好后再首次
# import cv2，其 GUI 后端初始化会挂死 —— 实测卡在第一帧的窗口创建上。
import cv2  # noqa: E402
import mujoco  # noqa: E402
import torch  # noqa: E402

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402

# ==================== 配置区 ====================

H1_ROOT = Path(__file__).resolve().parents[1]          # .../H1_build
# 默认用 v3 模型（101 回合数据训练）；想对照旧版时显式传
#   --checkpoint .../output_smolvla_h1_v2/checkpoints/last/pretrained_model   (60 回合, 4/10)
#   --checkpoint .../output_smolvla_h1/checkpoints/last/pretrained_model      (33 回合, 左臂未动)
DEFAULT_CHECKPOINT = H1_ROOT / "output_smolvla_h1_v3" / "checkpoints" / "last" / "pretrained_model"
MJCF_PATH = H1_ROOT / "mujoco" / "H1_scene.xml"
FPS = 30
IMAGE_SIZE = 512
ROBOT_TYPE = "h1_omnipicker"   # 与数据集 info.json 一致（仅元信息，策略不消费）

CUBE_BODY_NAMES = ("red_cube_left", "red_cube_right")
BIN_CENTER_XY = (0.7, 0.0)     # green_bin body 位置
BIN_XY_TOL = 0.03              # 内腔半宽 0.045，容差收紧到 0.03（贴墙不算）
BIN_Z_RANGE = (0.646, 0.75)    # 底面 0.646，壁顶 0.741，方块静止约 0.666

# 腕部小窗标题与遥操作完全一致（数采时的观感）
WRIST_VIEWER_TITLES = {
    "left_wrist_rgb": "H1 left wrist RGB",
    "right_wrist_rgb": "H1 right wrist RGB",
}

# 关节/夹爪/相机常量直接从数采记录器 import —— 单一事实来源，杜绝顺序漂移
sys.path.insert(0, str(H1_ROOT / "teleop"))
from h1_dataset_recorder import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_TASK,
    GRIPPER_ACTUATOR_NAMES,
    UPPER_BODY_JOINTS,
)

# 相机名 → 策略观测键（与训练 RENAME_MAP 语义一致：camera1=左腕 camera2=右腕 camera3=头）
CAMERA_TO_POLICY_KEY = {
    "left_wrist_rgb": "observation.images.camera1",
    "right_wrist_rgb": "observation.images.camera2",
    "head_rgb": "observation.images.camera3",
}

def show_wrist_windows(images: dict[str, np.ndarray]) -> bool:
    """刷新左右腕两个小窗（与遥操作数采时同款窗口）。

    画面直接复用本帧观测渲染结果（512×512 uint8 RGB），零额外渲染开销。
    窗口本体在 main() 里随 viewer 一起预先创建（teleop 同款流程）。
    返回 True 表示用户在腕部窗口按了 q（提前结束本回合）。
    """
    for camera_name, title in WRIST_VIEWER_TITLES.items():
        bgr = cv2.cvtColor(images[CAMERA_TO_POLICY_KEY[camera_name]], cv2.COLOR_RGB2BGR)
        cv2.imshow(title, bgr)
    # 处理 OpenCV 窗口事件（遥操作同款：单次 waitKey 服务两个窗口）
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def close_wrist_windows() -> None:
    for title in WRIST_VIEWER_TITLES.values():
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass  # 用户可能已手动关掉某个腕部窗口


# ==================== MuJoCo 侧工具 ====================


class H1RolloutEnv:
    """封装名称索引 + 观测渲染 + 动作执行 + 回合重置，全部复刻 recorder 的口径。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise ValueError("H1_scene.xml 缺少 'home' keyframe")
        self.home_id = home_id
        self.home_ctrl = model.key_ctrl[home_id].copy()

        # 与 recorder 相同的四组索引
        self.joint_qposadr = []
        self.joint_actuators = []
        for joint_name in UPPER_BODY_JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_position")
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"模型缺少关节/执行器: '{joint_name}'")
            self.joint_qposadr.append(int(model.jnt_qposadr[joint_id]))
            self.joint_actuators.append(actuator_id)
        self.gripper_actuators = []
        for actuator_name in GRIPPER_ACTUATOR_NAMES.values():  # left 在前，right 在后
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                raise ValueError(f"模型缺少夹爪执行器: '{actuator_name}'")
            self.gripper_actuators.append(actuator_id)
        self.camera_ids = {}
        for camera_name in CAMERA_NAMES:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"模型缺少相机: '{camera_name}'")
            self.camera_ids[camera_name] = camera_id
        self.cube_body_ids = {}
        for cube_name in CUBE_BODY_NAMES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
            if body_id < 0:
                raise ValueError(f"模型缺少方块 body: '{cube_name}'")
            self.cube_body_ids[cube_name] = body_id

        # 渲染器惰性创建：观看模式必须等 viewer 起来再建 GL 上下文（recorder 同款教训）
        self.renderers: dict[str, mujoco.Renderer] | None = None

    def ensure_renderers(self) -> None:
        if self.renderers is None:
            self.renderers = {
                name: mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
                for name in CAMERA_NAMES
            }

    def reset_episode(self, settle_seconds: float) -> None:
        """整场景回 home keyframe + 静置：方块从 keyframe 位姿落稳，ctrl 保持 home。"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        mujoco.mj_forward(self.model, self.data)
        settle_end = self.data.time + settle_seconds
        while self.data.time < settle_end - 1e-9:
            mujoco.mj_step(self.model, self.data)

    def get_state(self) -> np.ndarray:
        """20 维 state，与 recorder.record_frame 完全一致（夹爪用 actuator_length）。"""
        state = np.empty(20, dtype=np.float32)
        for i, address in enumerate(self.joint_qposadr):
            state[i] = self.data.qpos[address]
        for i, actuator_id in enumerate(self.gripper_actuators):
            state[18 + i] = self.data.actuator_length[actuator_id]
        return state

    def render_images(self) -> dict[str, np.ndarray]:
        """三相机 uint8 HWC 图像，键为策略观测键（cameraN）。"""
        self.ensure_renderers()
        images = {}
        for camera_name, renderer in self.renderers.items():
            renderer.update_scene(self.data, camera=self.camera_ids[camera_name])
            images[CAMERA_TO_POLICY_KEY[camera_name]] = renderer.render()
        return images

    def apply_action(self, action: np.ndarray) -> None:
        """20 维动作写 ctrl（关节位置目标 + 夹爪开合），限幅到 ctrlrange。"""
        for i, actuator_id in enumerate(self.joint_actuators):
            self._write_ctrl(actuator_id, action[i])
        for i, actuator_id in enumerate(self.gripper_actuators):
            self._write_ctrl(actuator_id, action[18 + i])

    def _write_ctrl(self, actuator_id: int, value: float) -> None:
        lo, hi = self.model.actuator_ctrlrange[actuator_id]
        self.data.ctrl[actuator_id] = np.clip(value, lo, hi)

    def cube_in_bin(self, cube_name: str) -> bool:
        pos = self.data.xpos[self.cube_body_ids[cube_name]]
        return (
            abs(pos[0] - BIN_CENTER_XY[0]) < BIN_XY_TOL
            and abs(pos[1] - BIN_CENTER_XY[1]) < BIN_XY_TOL
            and BIN_Z_RANGE[0] < pos[2] < BIN_Z_RANGE[1]
        )

    def cube_positions(self) -> dict[str, np.ndarray]:
        return {name: self.data.xpos[body_id].copy() for name, body_id in self.cube_body_ids.items()}


# ==================== 主循环 ====================


def run_episode(
    env: H1RolloutEnv,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    device: torch.device,
    task: str,
    max_steps: int,
    settle: float,
    live: bool,
    viewer=None,
) -> dict:
    env.reset_episode(settle)
    policy.reset()  # 清空动作队列与内部状态，每回合从零开始
    if viewer is not None and viewer.is_running():
        viewer.sync()

    frame_duration = 1.0 / FPS
    ever_in = {name: False for name in CUBE_BODY_NAMES}
    last_action = env.home_ctrl.copy()

    print(f"  [ep] 开始闭环（最多 {max_steps} 帧 ≈ {max_steps / FPS:.0f}s）")
    for step in range(max_steps):
        # 1) 观测：图像 + state 同一仿真时刻（与数采口径一致）
        images = env.render_images()
        if live and show_wrist_windows(images):
            print(f"  [ep] 第 {step} 帧：腕部窗口按 q，提前结束本回合")
            break
        obs: dict = {**images, "observation.state": env.get_state()}
        obs = prepare_observation_for_inference(obs, device, task, ROBOT_TYPE)
        obs = preprocessor(obs)

        # 2) 推理：select_action 内部按 chunk 弹队列，每帧只拿一步
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = postprocessor(action)
        action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)

        if not np.all(np.isfinite(action_np)):  # NaN/Inf 保护：沿用上一步动作
            print(f"  [ep] 第 {step} 帧动作含非有限值，沿用上一步")
            action_np = last_action
        last_action = action_np

        # 3) 执行 1/30s（时间基准推进，与 teleop 调度同模式）
        env.apply_action(action_np)
        target = env.data.time + frame_duration
        while env.data.time < target - 1e-9:
            mujoco.mj_step(env.model, env.data)

        # 4) 成功判定 + 显示
        for cube_name in CUBE_BODY_NAMES:
            if env.cube_in_bin(cube_name):
                ever_in[cube_name] = True
        end_in = {name: env.cube_in_bin(name) for name in CUBE_BODY_NAMES}
        if all(end_in.values()):
            print(f"  [ep] 第 {step} 帧：双块均已入桶，回合成功 ✓")
            break

        if viewer is not None and viewer.is_running():
            viewer.sync()
        if live:  # 观看模式按 30Hz 实时节奏回放；无头评估全速跑
            wall_target = (step + 1) * frame_duration
            overrun = time.monotonic() - run_episode._wall_start - wall_target
            if overrun < 0:
                time.sleep(-overrun)

        if (step + 1) % 150 == 0:
            cubes = env.cube_positions()
            cube_str = "  ".join(
                f"{name}: x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}"
                for name, pos in cubes.items()
            )
            print(f"  [ep] {step + 1}/{max_steps} 帧 | {cube_str}")

    end_in = {name: env.cube_in_bin(name) for name in CUBE_BODY_NAMES}
    return {
        "steps": step + 1,
        "ever_in": ever_in,
        "end_in": end_in,
        "success": all(end_in.values()),
        "partial": any(end_in.values()),
        "final_cubes": env.cube_positions(),
    }


def main() -> None:
    checkpoint = Path(ARGS.checkpoint) if ARGS.checkpoint else DEFAULT_CHECKPOINT
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"找不到 checkpoint 目录: {checkpoint}")
    task = ARGS.task or DEFAULT_TASK
    device = torch.device(ARGS.device)
    mode = "无头评估（全速）" if ARGS.headless else "仿真窗口 + 腕部小窗（30Hz 实时）"
    print(f"[load] checkpoint : {checkpoint}")
    print(f"[load] task       : {task}")
    print(f"[load] device     : {device} | 模式: {mode} | 回合数: {ARGS.episodes}")

    # 1) 策略 + 前后处理器（与 SO101 推理脚本同一调用序列）
    policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path=str(checkpoint)).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    image_keys = [k for k in policy.config.input_features if "image" in k]
    print(f"[load] 策略输入图像键: {image_keys}")  # 应为 camera1/2/3 —— 若是数据集原名说明 checkpoint 不对
    if sorted(image_keys) != sorted(CAMERA_TO_POLICY_KEY.values()):
        raise RuntimeError(
            f"checkpoint 图像键 {image_keys} 与推理组装键 {list(CAMERA_TO_POLICY_KEY.values())} 不一致，"
            "请确认用的是带 rename_map 训练出的 checkpoint"
        )

    # 2) MuJoCo 场景
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    env = H1RolloutEnv(model, data)
    print(f"[load] 场景: {MJCF_PATH.name} (nq={model.nq}, nu={model.nu}, timestep={model.opt.timestep})")

    # 3) 逐回合闭环（观看模式：主窗口相机固定为头部第一人称，同遥操作）
    results = []
    viewer_ctx = None
    try:
        if not ARGS.headless:
            import mujoco.viewer as mujoco_viewer

            viewer_ctx = mujoco_viewer.launch_passive(model, data)
            viewer_ctx.__enter__()
            viewer_ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer_ctx.cam.fixedcamid = env.camera_ids["head_rgb"]
            env.ensure_renderers()  # viewer 起来之后再建离屏渲染器（GL 上下文顺序）
            for title in WRIST_VIEWER_TITLES.values():  # 腕部小窗随 viewer 一起建好
                cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            print("[view] MuJoCo 窗口（头部视角）+ 左右腕小窗已启动；腕部窗口按 q 跳过本回合，Ctrl+C 退出")
        for ep in range(ARGS.episodes):
            print(f"\n===== 回合 {ep + 1}/{ARGS.episodes} =====")
            run_episode._wall_start = time.monotonic()
            result = run_episode(
                env, policy, preprocessor, postprocessor, device, task,
                max_steps=ARGS.max_steps, settle=ARGS.settle,
                live=not ARGS.headless, viewer=viewer_ctx,
            )
            results.append(result)
            cubes = result["final_cubes"]
            cube_str = "  ".join(
                f"{name}: ({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})" for name, pos in cubes.items()
            )
            print(
                f"  [ep] 结束：{result['steps']} 帧 | 结束入桶 L={'✓' if result['end_in']['red_cube_left'] else '✗'} "
                f"R={'✓' if result['end_in']['red_cube_right'] else '✗'} | "
                f"全程曾入桶 L={'✓' if result['ever_in']['red_cube_left'] else '✗'} "
                f"R={'✓' if result['ever_in']['red_cube_right'] else '✗'}\n"
                f"  [ep] 末态方块 {cube_str}"
            )
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
        if not ARGS.headless:
            close_wrist_windows()

    # 4) 汇总
    n = len(results)
    n_success = sum(r["success"] for r in results)
    n_partial = sum(r["partial"] for r in results)
    n_ever = sum(all(r["ever_in"].values()) for r in results)
    print("\n===== 成功率汇总 =====")
    print(f"回合数 {n} | 完全成功（双块入桶@结束） {n_success}/{n} = {n_success / n:.0%}")
    print(f"        | 部分成功（至少一块@结束） {n_partial}/{n} = {n_partial / n:.0%}")
    print(f"        | 全程曾双块入桶 {n_ever}/{n} = {n_ever / n:.0%}")
    print("提示：完全为 0 且部分成功 > 0，通常意味着单臂学会了、另一臂没学会，可回看末态方块坐标定位。")


if __name__ == "__main__":
    main()
