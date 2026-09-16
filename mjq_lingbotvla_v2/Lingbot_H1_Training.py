#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lingbot_H1_Training.py —— H1 仿真数据 后训练(post-training)纯 Python 启动器
====================================================================
【这个脚本做什么】
  用 H1 仿真数采数据(LeRobot v3.0,批次见下方配置区)微调 LingBot-VLA 2.0。
  等价于 CLI: CUDA_VISIBLE_DEVICES=0 bash train.sh tasks/vla/train_lingbotvla.py ...

【怎么调用的】★ 纯 Python 进程内调用(import),不是 subprocess 命令行
  和 Lingbot_Dataset_Creation.py 同一套路:
    ① 设 LOCAL_RANK/RANK/WORLD_SIZE/MASTER_* 环境变量 —— 模拟"单卡 torchrun"
       (train_lingbotvla.py 的 main() 里 dist.init_process_group(backend="nccl")
        按 env:// 协议读这些变量;arguments.py 的 local_rank 也是 os.getenv)
    ② 把参数填进 sys.argv(train_lingbotvla 用 parse_args 从 sys.argv 读配置)
    ③ 直接 import 项目训练入口的 main 函数调用 —— 同一进程跑完整个训练

【前置条件(缺一不可)】
  1. conda activate lingbotv2                 # 本机新建的环境(torch2.8.0+flash-attn)
  2. assets/norm_stats/h1.json 已生成          # 由 compute_norm_stats 算出(配套 h1_norm.yaml)
  3. models/ 下三份权重就位:
       lingbot-vla-v2-6b/(主模型)  Qwen3-VL-4B-Instruct/(tokenizer)
       (首跑关蒸馏,MoGe/depth/dino_video 暂不需要)

【产出】output/h1/ 下的 checkpoint(开环评估/deploy 都用它)

【怎么跑】
  ★ VSCode(推荐):打开本文件,点右上角 ▶ 运行(或 F5 调试,配置见 .vscode/launch.json)
    —— 脚本自带"解释器自纠正":就算 VSCode 当前选的是 base 环境,也会自动
       切到 lingbotv2 的 python 重启自己,不会再报 No module named 'numpy'
  ★ 终端:
    conda activate lingbotv2
    python mjq_lingbotvla_v2/Lingbot_H1_Training.py
====================================================================
"""

import os
import sys
from pathlib import Path

# ============ 解释器自纠正保险(VSCode/终端无论用哪个 python 启动都能跑) ============
# 现象:用 base 环境的 python 启动本脚本会报 ModuleNotFoundError: numpy
# 对策:检测到不是 lingbotv2 环境时,自动换成正确的解释器重启自己(os.execv 原地替换进程)
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}")
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# ====================================================================================

# ==================== 配置(改成你的参数)====================
# ★ 当前指向 v4_w3 批次(方案A:专家-only 续训 120k→200k,2026-09-02 起):
#   v4_w2@120000 闭环仍不抓取;home 探针定案=右首块幅值 11-20%、L_J6 超前
#   0.17、方向不一致("发起/终段欠拟合+边界超前累积"三层机制,详见
#   《V4闭环失败诊断.md》九/十章)。数据/环境/协议三审计全无罪;全模型
#   配方(B/C)用户叫停。本批次 = w2 配方逐字不动,再压 80000 步赌"曝光
#   不足"(依据:v4→v4_w 加权已把左发起 0→52%,曝光→学习因果成立)。
#   ★ 逐档门指标:每 8000 步落档后跑 home 探针(幅值≥10cm/L_J6<0.05/
#     方向→右块)。128k(~8h)不抬头、136k 仍平 → 判配方封顶,提前停。
#   ★★ 128000 档落盘后注释 h1_v4_w3.yaml 的 load_checkpoint_path:
#       中断重启直接再跑本脚本即可(enable_resume 自动找最新档)。
#   要切回 v4_w2 批次对比时改回:
#     CONFIG_YAML = "configs/vla/real_robot/h1_v4_w2.yaml"
#     OUTPUT_DIR  = "output/h1_v4_w2"
CONFIG_YAML = "configs/vla/real_robot/h1_v4_w3.yaml"  # ★ v4_w3:续训(load 自 h1_v4_w2@120000)
DATA_NAME = "h1"                                 # robot_config 名(configs/robot_configs/h1.yaml)
DATA_PATH = "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"   # 数据不变(同一份!)
NORM_STATS = "assets/norm_stats/h1_v4.json"      # 统计不变(同一份数据必须同一份统计!)
OUTPUT_DIR = "output/h1_v4_w3"                   # ★ 新目录;h1_v4/w/w2 的终档都不动
MASTER_PORT = "29616"                            # 避开 v3(29611)/v4(29612)/w(29613)/w2(29614)/full(29615)
# =============================================================

# 项目根目录(本脚本在 mjq_lingbotvla_v2/ 下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    # 切到项目根 + 加入 path(训练代码依赖相对路径和项目 import)
    os.chdir(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "tasks" / "vla"))   # 为了 import train_lingbotvla

    # ---- 设分布式环境变量(单卡模拟 torchrun;train.sh+torchrun 平时就是注入这些)----
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # 用 0 号卡(5090)
    os.environ["LOCAL_RANK"] = "0"                       # 机内卡号
    os.environ["RANK"] = "0"                             # 全局进程号
    os.environ["WORLD_SIZE"] = "1"                       # 总进程数(单卡=1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"              # 主节点(单机就是本机)
    os.environ["MASTER_PORT"] = MASTER_PORT

    # ---- 离线开关(与 train.sh 保持一致:全本地权重,不去 HF Hub 联网)----
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 显存防碎片
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ---- 设置 sys.argv ----
    # 说明:train_lingbotvla 的 parse_args 从 sys.argv 读配置
    #       (yaml 打底 + --xxx 覆盖;argparse 的 --a.b=c 形式)
    sys.argv = [
        "train_lingbotvla.py",
        str(PROJECT_ROOT / CONFIG_YAML),
        f"--data.data_name={DATA_NAME}",
        f"--data.train_path={str((PROJECT_ROOT / DATA_PATH).resolve())}",
        f"--data.norm_stats_file={NORM_STATS}",
        f"--train.output_dir={OUTPUT_DIR}",
    ]

    print("=" * 64)
    print("LingBot-VLA 2.0 后训练(H1 仿真数据)")
    print(f"  配置:   {CONFIG_YAML}")
    print(f"  数据:   {DATA_PATH}")
    print(f"  归一化: {NORM_STATS}")
    print(f"  输出:   {OUTPUT_DIR}")
    print("=" * 64)

    # ---- 进程内调用训练入口(与 torchrun 拉起后走的代码完全相同)----
    from train_lingbotvla import main as train_main
    train_main()


if __name__ == "__main__":
    main()


# =====================================================================
# 【等价的 CLI 训练命令】(效果与本脚本完全相同;多卡时用这种写法,
#   train.sh 会自动探测 GPU 数;NNODES/NPROC_PER_NODE 环境变量可覆盖)
#
#   conda activate lingbotv2
#   cd /home/mjq/robot_item/lingbot-vla-v2-main
#   CUDA_VISIBLE_DEVICES=0 bash train.sh tasks/vla/train_lingbotvla.py \
#       ./configs/vla/real_robot/h1.yaml \
#       --data.data_name h1 \
#       --data.train_path /home/mjq/robot_item/lingbot-vla-v2-main/H1_simulation_model/H1_build/data/h1_build_pick_v2 \
#       --data.norm_stats_file assets/norm_stats/h1.json \
#       --train.output_dir output/h1
#
# 【训练完成后的下一步】
#   开环评估(拿 checkpoint 在验证数据上对比动作):
#   python scripts/open_loop_eval.py \
#       --model_path output/h1/你的checkpoint --robo_name h1 \
#       --data_path /home/mjq/robot_item/lingbot-vla-v2-main/H1_simulation_model/H1_build/data/h1_build_pick_v2 \
#       --use_length 50
# =====================================================================
