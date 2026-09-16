#!/home/mjq/miniconda3/envs/xr-robotics/bin/python
# -*- coding:utf-8 -*-

'''H1 仿真 —— PICO 遥操作数采启动器（X 键开/保存回合）'''

"""
run_record.py
直接运行本文件 = 启动带录制的 PICO 遥操作，数据自动落下方 DATASET_DIR。

    ./run_record.py                # 直接执行（shebang 已指向 xr-robotics 环境）
    ./run_record.py --debug-xr     # 额外参数原样透传给 teleop

每回合节奏（按键）：
    A 重置方块到固定点（=v3 老点 0.65, ±0.30；定义在 H1_scene.xml
      的 home keyframe，rollout --spawn home 与此同源）
      → 等 1s 落稳 → X 开始录制 → 完成抓放入桶 → X 保存 → 循环
    * 录制中按 A = 丢弃当前回合（防方块瞬移污染数据）
    * 按 B = 双臂回 home（录制中按 B 自动保存）
    * 进近纪律与终末慢速要求见同目录《V4_数采操作说明.md》——这是 v4 数据质量的核心

内部等价于：
    conda activate xr-robotics
    cd H1_build && python teleop/h1_pico_teleop.py --record \\
        --repo-id <REPO_ID> --dataset-dir <DATASET_DIR> <透传参数>
"""

import os
import sys
from pathlib import Path

# ==================== 配置区 ====================
# 当前采 v4 数据集。v4 = v3 的斜坡夹爪约定（GRIPPER_COMMAND_SPEED=2.0，不变）
#  + 方块固定出生点沿用 v3 老点 (0.65, ±0.30)（H1_scene.xml home keyframe；
#    曾试过双臂之间的 (0.55,±0.12) 和随机带，均按用户要求改回/弃用）
#  + 头部完全固定（home 姿态 yaw=0/俯仰0.53 正对桌面，头显不再联动头部）
#  + 进近纪律/终末慢速协议（《V4_数采操作说明.md》）。
# v1~v3 的目录绝不能续采混批；目录不存在自动新建，存在自动续采。
# 重训前还要：trim 拷贝 → 重新生成 norm_stats → 训练配置数据集路径同步改。
DATASET_DIR = "data/h1_build_pick_v4"
REPO_ID = "mjq/h1_build_pick_v4"
TASK = ""    # 留空 = 用 teleop 默认任务语句；要改写这里，如 "Pick up ..."
# ===============================================

H1_ROOT = Path(__file__).resolve().parent                       # .../H1_build
PYTHON = "/home/mjq/miniconda3/envs/xr-robotics/bin/python"     # 遥操作/数采专用环境
TELEOP = H1_ROOT / "teleop" / "h1_pico_teleop.py"

if __name__ == "__main__":
    args = ["--record", "--repo-id", REPO_ID, "--dataset-dir", DATASET_DIR]
    if TASK:
        args += ["--task", TASK]
    os.chdir(H1_ROOT)  # DATASET_DIR 相对路径以 H1_build 为基准
    os.execv(PYTHON, [PYTHON, str(TELEOP), *args, *sys.argv[1:]])





# bash /opt/apps/roboticsservice/runService.sh
