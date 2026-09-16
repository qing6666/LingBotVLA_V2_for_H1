#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lingbot_Dataset_Creation.py —— SO101 数据集制作(算归一化统计 → so101.json)
====================================================================
【这个脚本做什么】
  扫描 dual_arm_dataset,算出 arm.position / effector.position 的归一化统计
  (mean/std),存成 so101.json。训练前必须的一步。

【怎么调用的】★ 纯 Python 进程内调用(import),不是 subprocess 命令行
  直接 import 项目自带的 compute_norm_stats.main 函数,在同一个 Python 进程里调用它。
  核心算法(RunningStats 流式统计)用项目已实现好的,不重写。

【产出】assets/norm_stats/so101.json(训练 + 推理共用,必须同一份)

【怎么跑】
  conda activate lingbotvla              # ★ 必须(要 lerobot/pyarrow)
  python mjq_lingbotvla_v2/Lingbot_Dataset_Creation.py
====================================================================
"""

import os
import sys
from pathlib import Path

# ==================== 配置(改成你的参数)====================
CONFIG_YAML = "configs/vla/norm_compute/so101_norm.yaml"  # ★ norm 专用(无 model 段,compute_norm_stats 用)
DATA_NAME = "so101"                                     # robot_config 名
DATA_PATH = "dual_arm_dataset"                          # LeRobot 数据集目录
NORM_PATH = "assets/norm_stats/so101.json"              # 输出 norm_stats
# =============================================================

# 项目根目录(本脚本在 mjq_lingbotvla_v2/ 下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    # 切到项目根 + 加入 path(compute_norm_stats 依赖相对路径和项目 import)
    os.chdir(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

    # ---- 设分布式环境变量(单卡模拟 torchrun;TrainingArguments 要 LOCAL_RANK 等)----
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    # ---- 设置 sys.argv ----
    # 说明:compute_norm_stats 内部用 parse_args 从 sys.argv 读配置
    #       (支持 yaml + --xxx 覆盖)。我们把参数填进 sys.argv,
    #       然后直接 import 它的 main 函数调用 —— 整个过程在同一个 Python 进程里,
    #       不是开子进程跑命令行。
    sys.argv = [
        "compute_norm_stats.py",
        str(PROJECT_ROOT / CONFIG_YAML),
        f"--data.data_name={DATA_NAME}",
        f"--data.train_path={str((PROJECT_ROOT / DATA_PATH).resolve())}",
        f"--data.norm_path={NORM_PATH}",
    ]

    print("=" * 64)
    print("  SO101 数据集制作(import 调用 compute_norm_stats,同进程)")
    print("=" * 64)
    print(f"  配置    : {CONFIG_YAML}")
    print(f"  robot   : {DATA_NAME}(→ configs/robot_configs/{DATA_NAME}.yaml)")
    print(f"  数据集  : {DATA_PATH}")
    print(f"  输出    : {NORM_PATH}")
    print("=" * 64)

    # ★ 纯 Python 进程内运行(runpy,不是 subprocess 命令行)
    # 说明:compute_norm_stats 的逻辑写在 `if __name__=="__main__"` 块里(没有独立 main 函数),
    # 所以用 runpy.run_path 以 "__main__" 方式运行它,触发该块执行。
    # sys.argv 已在上面设好,它会:读配置 → 加载数据 → 按 so101.yaml 映射 → RunningStats 统计 → 存 so101.json
    import runpy
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "compute_norm_stats.py"), run_name="__main__")

    out = PROJECT_ROOT / NORM_PATH
    if out.exists():
        print(f"\n✅ 完成!norm_stats 已生成:{NORM_PATH}")
        print("   下一步:用 so101.json 去训练(train_lingbotvla)。")
    else:
        print(f"\n❌ 未找到输出 {NORM_PATH},看上面的报错。")


if __name__ == "__main__":
    main()
