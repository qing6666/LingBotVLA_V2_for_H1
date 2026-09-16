#!/usr/bin/env python3
"""模型加载与查看器 —— H1_build 通用工具。

用法:
  python scripts/view.py                        # 查看 mujoco/H1_scene.xml (最终成品)
  python scripts/view.py mujoco/xxx.xml         # 查看指定模型 (如中间产物 H1.xml)
  python scripts/view.py --stats                # 只打印模型信息, 不开窗口(无界面验证用)

程序做四件事:
  1. 加载模型:  MjModel.from_xml_path() 解析+编译 MJCF (有问题这里就会报错)
  2. 分配状态:  MjData 存放 qpos/qvel/ctrl 等运行时数据
  3. 初始化:    优先用模型里的 "home" keyframe; 没有就把关节设到行程中点
  4. 打开窗口:  mujoco.viewer.launch() 交互式查看
                (鼠标左键旋转 / 右键平移 / 滚轮缩放 / 空格暂停 / Ctrl+拖 施力)
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = HERE / "mujoco/H1_scene.xml"   # 默认看最终成品


def print_stats(model: mujoco.MjModel) -> None:
    """打印模型的核心数字 —— 每次改完模型都应对照检查这张表。"""
    print("── 模型信息 ─────────────────────────────")
    print(f"  nq={model.nq:<3d} (广义位置/关节数)   nv={model.nv:<3d} (速度数)")
    print(f"  nu={model.nu:<3d} (执行器数)          nbody={model.nbody:<3d} (刚体数)")
    print(f"  nmesh={model.nmesh:<3d} (网格数)      nkey={model.nkey} (keyframe数)")

    print("  ── 关节 ──")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"    {name:12s} range=[{model.jnt_range[j][0]:+.4f}, "
              f"{model.jnt_range[j][1]:+.4f}]")

    if model.nu:
        print("  ── 执行器 ──")
        for a in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            print(f"    [{a:2d}] {name}  ctrlrange={model.actuator_ctrlrange[a]}")


def init_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """初始化关节状态: 有 home keyframe 用它, 没有就取各关节行程中点。"""
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        print("── 已复位到 home keyframe ──")
        return

    # 没有 keyframe: 把每个关节放到行程中点 (比如肘部[0,2.09] -> 1.05, 手臂微弯)
    for j in range(model.njnt):
        lo, hi = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = 0.5 * (lo + hi)
    print("── 无 home keyframe, 关节已设到行程中点 ──")


def main() -> None:
    parser = argparse.ArgumentParser(description="H1_build 模型查看器")
    parser.add_argument("model", nargs="?", default=str(DEFAULT_MODEL),
                        help="MJCF 模型路径 (默认 mujoco/H1.xml)")
    parser.add_argument("--stats", action="store_true",
                        help="只打印模型信息, 不打开窗口")
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        # 相对路径兜底: 按脚本位置再找一次 (在别的目录下启动也不怕)
        fallback = HERE / args.model
        if fallback.exists():
            path = fallback
        else:
            raise SystemExit(f"模型不存在: {path}")

    # ① 加载  ② 分配状态
    print(f"加载模型: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)

    print_stats(model)

    if args.stats:
        return

    # ③ 初始化 + 前向计算(更新坐标系/惯量等派生量, 否则渲染会用旧数据)
    init_state(model, data)
    mujoco.mj_forward(model, data)

    # ④ 打开交互窗口 (关掉窗口程序才退出)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
