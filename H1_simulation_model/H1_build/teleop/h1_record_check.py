#!/usr/bin/env python3
"""无头验证 H1DatasetRecorder 的完整数采链路（不需要 PICO / XRoboToolkit / 显示器）。

做什么
--------
1. 加载 ``mujoco/H1_scene.xml``, 脚本化驱动双臂（正弦关节运动 + 夹爪开合）；
2. 按 30fps 仿真时刻调度 ``record_frame``, 采 N 个短 episode 并保存；
3. 用 LeRobotDataset 回读校验：帧数 / episode 数 / fps / features 维度 / 视频解码。

通过标准：全部 ✅, 退出码 0。这是 PICO 实机数采前的门禁 —— 记录模块在本机
验证通过后, 实机端只需把 ``h1_pico_teleop.py --record`` 跑起来, 记录代码零改动。

用法
--------
    cd H1_build
    python teleop/h1_record_check.py                 # 默认 2 回合 x 4 秒
    python teleop/h1_record_check.py --fresh         # 删掉旧验证数据集重采
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # 无头离屏渲染, 必须在 import mujoco 前设置

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

H1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = H1_ROOT / "mujoco" / "H1_scene.xml"
DEFAULT_ROOT = H1_ROOT / "data" / "h1_record_check"

# 脚本化运动用到的执行器（subset；其余执行器保持 home ctrl）
MOTION_JOINTS = {
    "Left_J4_position": 0.15,
    "Right_J4_position": 0.15,
    "Left_J6_position": 0.10,
    "Right_j6_position": 0.10,
}
MOTION_PERIOD = 3.0  # 秒


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """回到 home keyframe（含方块位姿与 ctrl），并刷新派生量。"""
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)


def run_episode(recorder, model, data, seconds: float, fps: int, quiet: bool = False) -> int:
    """跑一个脚本化 episode：正弦手臂运动 + 夹爪开合，按 fps 记录，返回帧数。"""
    actuator_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name): amplitude
        for name, amplitude in MOTION_JOINTS.items()
    }
    gripper_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_omnipicker_gripper_opening"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_omnipicker_gripper_opening"),
    ]
    home_ctrl = data.ctrl.copy()

    recorder.start_episode()
    recorded = 0
    next_record_time = float(data.time)
    end_time = float(data.time) + seconds
    while float(data.time) < end_time:
        phase = 2.0 * np.pi * float(data.time) / MOTION_PERIOD
        for actuator_id, amplitude in actuator_ids.items():
            data.ctrl[actuator_id] = home_ctrl[actuator_id] + amplitude * np.sin(phase)
        for i, actuator_id in enumerate(gripper_ids):
            data.ctrl[actuator_id] = 0.5 + 0.5 * np.sin(phase + i * np.pi)  # 交替开合 1..0..1

        mujoco.mj_step(model, data)
        if float(data.time) >= next_record_time:
            if recorder.record_frame(data):
                recorded += 1
            next_record_time += 1.0 / fps
    recorder.save_episode()
    if not quiet:
        print(f"  仿真 {seconds}s, 记录 {recorded} 帧 (期望 ~{int(seconds * fps)})")
    return recorded


def verify_dataset(root: Path, repo_id: str, fps: int,
                   expect_episodes: int, expect_new_frames: int) -> bool:
    """回读校验：帧数/episode 数/fps/维度/视频文件/单帧解码。

    expect_* 允许数据集此前已存在（续采）：期望值 = 原有 + 本次新增。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "✅" if condition else "❌"
        ok = ok and condition
        print(f"  {mark} {label}{(' — ' + detail) if detail else ''}")

    dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
    check("总帧数 = meta.total_frames", len(dataset) == dataset.meta.total_frames,
          f"{len(dataset)} 帧")
    check("episode 数", dataset.meta.total_episodes == expect_episodes,
          f"{dataset.meta.total_episodes} / 期望 {expect_episodes}")
    check("fps", dataset.meta.fps == fps, str(dataset.meta.fps))
    state_shape = tuple(dataset.meta.features["observation.state"]["shape"])
    action_shape = tuple(dataset.meta.features["action"]["shape"])
    check("state 维度 20", state_shape == (20,), str(state_shape))
    check("action 维度 20", action_shape == (20,), str(action_shape))

    # v3.0 布局：视频按 chunk 文件存（多个 episode 合并进一个 mp4，
    # resume 续采会再开一组文件），真不变量是"每路相机至少有一个 mp4"。
    covered = {
        path.parent.parent.name.removeprefix("observation.images.")
        for path in (root / "videos").rglob("*.mp4")
    }
    check("三路相机都有 mp4", covered == {"head_rgb", "left_wrist_rgb", "right_wrist_rgb"},
          f"{sorted(covered)}")

    try:
        sample = dataset[0]
        image = sample["observation.images.head_rgb"]
        check("首帧图像可解码", tuple(image.shape) == (3, 512, 512), str(tuple(image.shape)))
        state = np.asarray(sample["observation.state"])
        check("state 有限且非全零", bool(np.all(np.isfinite(state))) and float(np.abs(state).sum()) > 1e-6,
              f"range [{state.min():.3f}, {state.max():.3f}]")
        check("task 字符串", isinstance(sample["task"], str) and len(sample["task"]) > 0,
              repr(sample["task"])[:60])
        last = dataset[len(dataset) - 1]
        check("末帧属于最后一个 episode",
              int(last["episode_index"]) == dataset.meta.total_episodes - 1,
              f"episode_index={int(last['episode_index'])}")
    except Exception as error:  # noqa: BLE001
        check("首帧读取", False, f"{type(error).__name__}: {error}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default="mjq/h1_record_check")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fresh", action="store_true", help="先删除已有验证数据集再采")
    args = parser.parse_args()

    if args.fresh and args.root.exists():
        shutil.rmtree(args.root)
        print(f"[check] 已删除旧数据集: {args.root}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from h1_dataset_recorder import H1DatasetRecorder

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    print(f"[check] 模型加载: nq={model.nq} nu={model.nu} timestep={model.opt.timestep}")

    recorder = H1DatasetRecorder(
        model, repo_id=args.repo_id, root=args.root, fps=args.fps,
        task="scripted motion check",
    )
    # 续采时基准值非零：期望值 = 原有 + 本次新增
    base_episodes = recorder.dataset.meta.total_episodes
    base_frames = recorder.dataset.meta.total_frames

    print(f"[check] 开始采集 {args.episodes} 个 episode x {args.seconds}s @ {args.fps}fps")
    frames = 0
    for _ in range(args.episodes):
        reset_to_home(model, data)
        mujoco.mj_step(model, data, nstep=100)  # 0.1s 稳定，不录制
        frames += run_episode(recorder, model, data, args.seconds, args.fps)
    recorder.stop()  # 内部先等后台编码完, 再 finalize

    print(f"\n[check] 回读校验 {args.root}")
    if not verify_dataset(args.root, args.repo_id, args.fps,
                          base_episodes + args.episodes, base_frames + frames):
        print("\n[check] ❌ 校验未通过")
        return 1
    print("\n[check] ✅ 全部通过 —— 记录链路可用, PICO 端直接 --record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
