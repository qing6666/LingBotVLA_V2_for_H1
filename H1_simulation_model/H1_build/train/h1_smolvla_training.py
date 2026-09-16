#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''SmolVLA H1 仿真数采数据训练程序'''

"""
h1_smolvla_training.py
SmolVLA H1 双臂仿真训练（纯 Python 版 —— 用 Python 对象构造配置，功能等价于 lerobot-train CLI）
参照 mjq_smolVLA/Dual_Arm_Smolvla_training.py（SO101 双臂）改写。

相机：MuJoCo 仿真三相机 head_rgb（头）+ left_wrist_rgb（左腕）+ right_wrist_rgb（右腕），512×512
数据：H1_build 数采 v2 —— 101 回合 / 78108 帧（两批续采：60 + 41，头部俯视姿态，双臂都有
      实质动作），state/action 20 维 = 上身 18 关节 + 左右夹爪，单任务语意
      "Pick up the red cube and put it into the green bin"（语言条件自动生效，无需额外配置）
      v1（33 回合旧姿态）模型在 output_smolvla_h1、v2（60 回合）模型在 output_smolvla_h1_v2，
      均保留可对照评估（v2 闭环成功率 4/10）。

与 SO101 版的唯一结构性差异——rename_map：
      smolvla_base 预训练的视觉特征键是 camera1/camera2/camera3（SO101 数据集键名恰好一致），
      H1 数据集键名是 head_rgb/left_wrist_rgb/right_wrist_rgb，必须重命名对齐后权重才能接上。
      对齐语义沿用 SO101：camera1=左腕, camera2=右腕, camera3=头。

运行：
    conda activate lerobot312
    cd ~/robot_item/lerobot-main
    python H1_simulation_model/H1_build/train/h1_smolvla_training.py

依赖的本地补丁（2026-08-19）：lerobot 0.5.2 的 src/lerobot/policies/factory.py 里
make_policy 已打补丁 —— rename_map 现在会同步作用于 input/output features 的键名。
没有它，策略按数据集原名找图像、batch 里却是改名后的 cameraN，第一步就报
"All image features are missing from the batch"。
若日后升级/重装 lerobot 后报同样的错，重新打这个补丁（见 H1_仿真学习笔记/记忆）。
"""

import sys
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.policies.factory import make_policy_config
from lerobot.scripts.lerobot_train import train


# ==================== 配置区 ====================

# ---- 数据集（H1 仿真数采 v2 目录续采后：101 回合 / 78108 帧，v3.0 校验全绿）----
DATASET_REPO_ID = "mjq/h1_build_pick_v2"
DATASET_ROOT = "/home/mjq/robot_item/lerobot-main/H1_simulation_model/H1_build/data/h1_build_pick_v2"

# ---- 特征重命名：H1 数据集键 → smolvla_base 预训练期望键 ----
# 语义对照 SO101 双臂：camera1=左腕 D405, camera2=右腕 D405, camera3=头部相机
RENAME_MAP = {
    "observation.images.left_wrist_rgb": "observation.images.camera1",
    "observation.images.right_wrist_rgb": "observation.images.camera2",
    "observation.images.head_rgb": "observation.images.camera3",
}

# ---- 预训练模型 ----
PRETRAINED_PATH = "lerobot/smolvla_base"    # HF Hub 上的 SmolVLA 预训练模型（微调起点）
POLICY_TYPE = "smolvla"                     # 20 维 state/action 与 3 相机由数据集自动推断

# ---- 输出（v3 新目录 = 101 回合全量重训；v2/v1 旧模型保留便于对照）----
OUTPUT_DIR = "/home/mjq/robot_item/lerobot-main/H1_simulation_model/H1_build/output_smolvla_h1_v3"
JOB_NAME = "smolvla_h1_v3"

# ---- 冒烟开关：True = 2000 步快速验证链路（输出目录自动换 _smoke），False = 正式训练 ----
SMOKE = False

# ---- 训练超参 ----
# 以下学习率/调度均沿用 v1 已验证配方（smolvla 代码默认，v1 训练 loss 2.28→0.03、
# 策略可闭环抓放入桶）：optimizer_lr=1e-4, weight_decay=1e-10, warmup=1000 步,
# 余弦退火到 2.5e-6。build_config 里只显式改 scheduler_decay_steps（见注释）。
#
# STEPS：总训练步数（每步处理一个 batch）。
#   现有 101 回合 / 78108 帧（v2 的 2.03 倍）；STEPS=50000 + BATCH_SIZE=8
#   → 看约 (50000×8)/78108 ≈ 5.1 遍（v1 7.2 遍、v2 6.2 遍；数据更多遍数略少，
#   且 scheduler_decay_steps 已跟随 STEPS，余弦退火在训练结束时恰好走完）。
#   预计耗时约 2.1 小时（实测每步 ≈0.153s）。
STEPS = 50000
# BATCH_SIZE：每次迭代同时喂给模型的样本数。
#   3 相机 512×512（RTX 5090 32G 跑 8 无压力）；OOM 降到 4。
#   保持 8 不动：学习率与 batch 耦合，v1 验证过这对组合。
BATCH_SIZE = 8
# NUM_WORKERS：数据加载进程数。3 路 AV1 视频解码较重，24 核机器给 8
#（v1 用默认 4 偏保守）；若 CPU 占用异常可回 4。
NUM_WORKERS = 8
# SAVE_FREQ：每隔多少步存一个 checkpoint（v1 只存了最终一步）。
#   50000 步存 5 份（10k~50k）：万一最终步过拟合，可回头评估中间档。
SAVE_FREQ = 10000
DEVICE = "cuda"            # 没有 GPU 改 "cpu"


# ==================== 构造训练配置（纯 Python 对象，等价 lerobot-train 的 CLI 参数）====================
def build_config() -> TrainPipelineConfig:
    steps = STEPS
    output_dir = OUTPUT_DIR
    job_name = JOB_NAME
    if SMOKE:
        steps = 2000
        output_dir = OUTPUT_DIR + "_smoke"
        job_name = JOB_NAME + "_smoke"
        print(f"[smoke] 冒烟模式：{steps} 步，输出 {output_dir}")

    # 1) Policy 配置：从 smolvla_base 预训练微调
    #    等价 CLI：--policy.type=smolvla --policy.path=lerobot/smolvla_base --policy.device=cuda
    policy = make_policy_config(
        POLICY_TYPE,
        pretrained_path=Path(PRETRAINED_PATH),   # 微调起点：smolvla_base 预训练权重
        device=DEVICE,
        push_to_hub=False,                       # 训练完不上传 HF Hub
    )
    # 余弦退火长度跟随总步数（smolvla 代码默认 30000，恰好等于正式步数；
    # 冒烟 2000 步时也同步缩短，避免"退火还没走完训练就结束"）。
    policy.scheduler_decay_steps = steps

    # 2) Dataset 配置：本地全量加载
    #    等价 CLI：--dataset.repo_id=... --dataset.root=... --dataset.streaming=false
    dataset = DatasetConfig(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        streaming=False,
    )

    # 3) 组装训练管线（等价其余 CLI 参数）
    cfg = TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        output_dir=Path(output_dir),
        job_name=job_name,
        batch_size=BATCH_SIZE,
        steps=steps,
        num_workers=NUM_WORKERS,                 # 3 路 AV1 解码用 8 进程加载
        save_freq=SAVE_FREQ,                     # 10k/20k/30k 三份 checkpoint
        wandb=WandBConfig(enable=False),         # 等价 --wandb.enable=false
        rename_map=RENAME_MAP,                   # H1 键名 → smolvla_base 的 cameraN
    )
    return cfg


def main():
    cfg = build_config()
    train(cfg)    # 官方训练入口，传入构造好的 TrainPipelineConfig
    # 训练完自动画 loss 曲线（中英文标注）→ output_dir/loss_curve.png
    try:
        sys.path.insert(0, "/home/mjq/robot_item/lerobot-main/mjq_smolVLA")  # plot_loss 所在目录
        from plot_loss import plot_loss_curve
        plot_loss_curve(cfg.output_dir)
    except Exception as e:
        print(f"[warning] 画 loss 曲线失败（不影响训练结果）：{e}")


if __name__ == "__main__":
    main()


# ==================== 等价命令行（lerobot-train CLI）====================
# 本脚本的纯 Python 配置，等价于下面这条 CLI 命令：
#
#   conda activate lerobot312
#   lerobot-train \
#       --policy.path=lerobot/smolvla_base \
#       --policy.type=smolvla \
#       --dataset.repo_id=mjq/h1_build_pick_v2 \
#       --dataset.root=/home/mjq/robot_item/lerobot-main/H1_simulation_model/H1_build/data/h1_build_pick_v2 \
#       --dataset.streaming=false \
#       --output_dir=/home/mjq/robot_item/lerobot-main/H1_simulation_model/H1_build/output_smolvla_h1_v3 \
#       --job_name=smolvla_h1_v3 \
#       --policy.device=cuda \
#       --wandb.enable=false \
#       --steps=50000 \
#       --batch_size=8 \
#       --num_workers=8 \
#       --save_freq=10000 \
#       --policy.scheduler_decay_steps=30000 \
#       --policy.push_to_hub=false \
#       --rename_map observation.images.left_wrist_rgb=observation.images.camera1 \
#       --rename_map observation.images.right_wrist_rgb=observation.images.camera2 \
#       --rename_map observation.images.head_rgb=observation.images.camera3
#
# 与 v1 / v2 的对照：
#   数据集   h1_build_pick     33 回合 / 22116 帧（头部平视，左臂几乎不动）
#          → h1_build_pick_v2  60 回合 / 38565 帧（头部俯视，双臂都有动作）
#          → h1_build_pick_v2 101 回合 / 78108 帧（同目录续采 41 回合）
#   STEPS   20000（7.2 遍）→ 30000（6.2 遍）→ 50000（5.1 遍；余弦退火同步走完）
#   输出    output_smolvla_h1 → _v2 → _v3（旧模型全部保留可对照）
#   其余    lr=1e-4 / warmup=1000 / batch=8 沿用 v1 验证过的配方不变
#   v2 闭环成绩：10 回合无头评估双块入桶 4/10（v1 左臂从未动过）。
