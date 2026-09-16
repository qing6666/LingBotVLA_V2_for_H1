#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lingbot_H1_Training_Full.py —— 方案B:全模型微调(VLM 解冻)启动器
====================================================================
【这个脚本做什么】
  与 Lingbot_H1_Training.py 完全同一套启动套路,唯一区别是指向
  configs/vla/real_robot/h1_v4_full.yaml:
    ★ train_expert_only: false —— 解冻整个 Qwen3-VL 一起训(官方配方),
      从官方底座 models/lingbot-vla-v2-6b 冷启动(全新优化器,步数从 0 计);
    ★ enable_fsdp_offload: true —— 权重/优化器驻 CPU 按需上卡
      (32GB 单卡装下 5.7B 可训练参数的唯一办法;代价:速度变慢,数值待实测);
    ★ lr 5e-5 + vit_lr 1e-6,global batch 1。
  ★ 前置补丁(2026-09-04 已打,共 3 个文件,备份/diff/复原命令见
    mjq_lingbotvla_v2/B_patch/README.md):
    补丁1 lingbotvla/distributed/torch_parallelize.py —— 放行"单卡 fsdp2 + offload";
    补丁2 tasks/vla/train_lingbotvla.py —— 单卡梯度裁剪绕开 NCCL-CPU;
    补丁3+4 lingbotvla/optim/muon.py —— muon 集合通信单卡恒等化 + NS 借 GPU 算。
    方案 B 不行,三条 cp 全部复原(命令在 README)。
  动机与差异清单见 h1_v4_full.yaml 文件头注释。
  证据链见项目根《V4闭环失败诊断.md》第九/十一章。

【试跑 vs 正跑】(本脚本支持命令行覆盖参数,追加在末尾,后写的赢)
  200 步试跑(验证四件事:不 OOM / 真实 it/s / loss 正常 / 能落 DCP 档;
  独立目录不污染正跑,跑完即删,~30-60 分钟):
    python mjq_lingbotvla_v2/Lingbot_H1_Training_Full.py \
        --train.max_steps=200 --train.save_steps=100 \
        --train.output_dir=output/h1_v4_full_TRIAL
  正跑 20000 步(★ 由用户亲自启动,试跑通过后再跑):
    python mjq_lingbotvla_v2/Lingbot_H1_Training_Full.py
  中断重启:直接再跑本脚本(enable_resume 自动找 output/h1_v4_full 最新档)。

【怎么跑】
  ★ VSCode:打开本文件点右上角 ▶(自带解释器自纠正,见下)
  ★ 终端:conda activate lingbotv2 && python mjq_lingbotvla_v2/Lingbot_H1_Training_Full.py
====================================================================
"""

import os
import sys
from pathlib import Path

# ============ 解释器自纠正保险(VSCode/终端无论用哪个 python 启动都能跑) ============
# 必须放在所有第三方 import 之前:用 base 环境的 python 启动本脚本会报
# ModuleNotFoundError: numpy,检测到不是 lingbotv2 环境时自动换解释器重启自己
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}")
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# ====================================================================================

# ==================== 配置(改成你的参数)====================
# ★ 方案B(2026-09-04 w3 终审"专家-only 封顶"后转正):全模型配方。
#   官方 real_robot.yaml/SO101 复现都不设 train_expert_only(默认 false=全模型),
#   我们 32GB 单卡此前被迫只训专家。w3 探针终审:152k 步(60.8 万样本)
#   发起幅值仍钉在门的 40-60%,L_J6 五档纹丝不动 → 冻 VLM 特征是唯一活口
#   假设。本批次换全模型配方,其余(数据/加权窗口/归一化/动作权重)与 w3
#   逐字一致 —— 变量只有配方。
#   权重起点 = 官方底座 models/lingbot-vla-v2-6b 冷启动(写死在
#   h1_v4_full.yaml 的 model.model_path,不用 load_checkpoint_path ——
#   换配方后旧优化器状态对不上;model_path 加载=权重恢复+全新优化器)。
CONFIG_YAML = "configs/vla/real_robot/h1_v4_full.yaml"  # ★ 全模型配方(v4_full)
DATA_NAME = "h1"                                 # robot_config 名(configs/robot_configs/h1.yaml)
DATA_PATH = "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"   # 数据不变(同一份!)
NORM_STATS = "assets/norm_stats/h1_v4.json"      # 统计不变(同一份数据必须同一份统计!)
OUTPUT_DIR = "output/h1_v4_full"                 # ★ 新目录;h1_v4/h1_v4_w/h1_v4_w2 全不动
MASTER_PORT = "29615"                            # 避开 v3(29611)/v4(29612)/v4_w(29613)/v4_w2(29614)
# =============================================================

# 项目根目录(本脚本在 mjq_lingbotvla_v2/ 下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    # 命令行透传:调用方追加的 --xxx=yyy 原样转给训练入口(后写的赢,可覆盖
    # yaml/本脚本的默认值)—— 显存试跑就是靠它注入 max_steps/save_steps/output_dir
    extra_args = list(sys.argv[1:])

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
    #       (yaml 打底 + --xxx 覆盖;argparse 的 --a.b=c 形式,后写的赢)
    sys.argv = [
        "train_lingbotvla.py",
        str(PROJECT_ROOT / CONFIG_YAML),
        f"--data.data_name={DATA_NAME}",
        f"--data.train_path={str((PROJECT_ROOT / DATA_PATH).resolve())}",
        f"--data.norm_stats_file={NORM_STATS}",
        f"--train.output_dir={OUTPUT_DIR}",
    ] + extra_args

    print("=" * 64)
    print("LingBot-VLA 2.0 后训练(H1 仿真数据 | ★全模型配方 v4_full)")
    print(f"  配置:   {CONFIG_YAML}")
    print(f"  数据:   {DATA_PATH}")
    print(f"  归一化: {NORM_STATS}")
    print(f"  输出:   {OUTPUT_DIR}")
    if extra_args:
        print(f"  覆盖:   {' '.join(extra_args)}")
    print("=" * 64)

    # ---- 进程内调用训练入口(与 torchrun 拉起后走的代码完全相同)----
    from train_lingbotvla import main as train_main
    train_main()


if __name__ == "__main__":
    main()
