#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lingbot_H1_NormStats.py —— H1 数据集 norm 统计量计算 纯 Python 启动器
=====================================================================
【这个脚本做什么】
  对一份 LeRobot 数据集流式扫描,算出各状态/动作特征的 mean/std,
  产出 assets/norm_stats/xxx.json —— 训练时 VLADataset 用它做归一化,
  评测/rollout 用同一份 json 反归一化。等价于 CLI:
    CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ...

【★什么时候需要跑:norm 统计跟着"数据"走,不跟着训练次数走】
  - 新采一批数据(如 v5)、裁剪方式或剔除回合变了 → 重算
  - 数据没动(中断重跑/续训/只改超参) → 不用重算,复用旧 json
  - 铁律:统计必须和训练 loader 实际读到的数据完全一致
    (先 trim_idle_prefix.py 裁剪、再对裁剪副本算统计;本脚本配置区
     的 DATA_PATH 必须与训练 yaml 里的 train_path 逐字一致)

【怎么调用的】★ 纯 Python 进程内执行(runpy),不是 subprocess:
  compute_norm_stats.py 的逻辑全写在 if __name__ == "__main__": 块里,
  没有 main() 函数可 import,所以用 runpy 以 __main__ 身份在当前进程跑。
  环境变量方面只需 LOCAL_RANK/RANK/WORLD_SIZE=0/0/1(arguments.py 的
  __post_init__ 强制读这三个;WORLD_SIZE=1 时脚本直接跳过 NCCL 初始化,
  不需要 MASTER_ADDR/PORT —— 这点和训练启动器不同)。

【算完之后】自动体检:扫描产出 json 里所有 std < 1e-3 的维度(恒定维陷阱)。
  v4 头部锁死就踩过这个坑:std 精确=0 → meanstd 归一化 0/0=NaN 直接报废
  训练,必须打 0.02 地板(见 mjq_lingbotvla_v2/fix_v4_head_std.py)。
  体检只读不写,发现问题会给出确切的处理指引。

【产出】assets/norm_stats/<版本>.json(约几 MB)

【怎么跑】
  ★ VSCode:打开本文件点右上角 ▶(解释器自纠正,选错环境也能跑)
    终端:
      conda activate lingbotv2
      python mjq_lingbotvla_v2/Lingbot_H1_NormStats.py             # 正式算
      python mjq_lingbotvla_v2/Lingbot_H1_NormStats.py --dry-run   # 只看参数不真算
  参考耗时:46 万帧 ≈ 80 分钟(5090 单卡)
=====================================================================
"""

import os
import sys
import runpy
import json
import argparse
from pathlib import Path

# ============ 解释器自纠正保险(VSCode/终端无论用哪个 python 启动都能跑) ============
# 现象:用 base 环境的 python 启动会报 ModuleNotFoundError
# 对策:检测到不是 lingbotv2 环境时,自动换成正确的解释器重启自己(os.execv 原地替换)
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}")
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# ====================================================================================

# ==================== 配置(改成你的参数)====================
# ★ 当前指向 v4 批次(302 episodes / 460499 帧 裁剪副本,统计已算过并打过头部
#   std 地板)。换新批次(如 v5)时四处一起改,且 DATA_PATH 必须与训练 yaml
#   的 train_path 逐字一致:
CONFIG_YAML = "configs/vla/norm_compute/h1_v4_norm.yaml"  # norm 计算专用配置
DATA_NAME = "h1"                                          # robot_config 名
DATA_PATH = "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"  # ★=训练用数据
NORM_JSON = "assets/norm_stats/h1_v4.json"                # 产出(训练/评测共用)
# v4 时的值(历史记录):
#   CONFIG_YAML = "configs/vla/norm_compute/h1_v4_norm.yaml"
#   DATA_PATH   = "H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed"
#   NORM_JSON   = "assets/norm_stats/h1_v4.json"
# =============================================================

# 项目根目录(本脚本在 mjq_lingbotvla_v2/ 下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 恒定维体检阈值:任何维度 std 低于它 = 数据里该维全程不动,须人工处理
STD_FLOOR_CHECK = 1e-3


def check_constant_dims(norm_json: Path):
    """算完统计后的自动体检:找出所有 std≈0 的维度(恒定维陷阱)。

    只读不写。发现问题打印处理指引后【不】自动修——地板值和涉及的键
    要由人确认(参照 mjq_lingbotvla_v2/fix_v4_head_std.py 的做法)。
    """
    if not norm_json.exists():
        print(f"[体检跳过] {norm_json} 不存在")
        return
    d = json.loads(norm_json.read_text())
    stats = d.get("norm_stats", {})
    suspicious = [
        f"{k}[{i}] std={s:.3g}"
        for k, v in sorted(stats.items())
        for i, s in enumerate(v.get("std", []))
        if s < STD_FLOOR_CHECK
    ]
    print("\n" + "=" * 64)
    if not suspicious:
        print(f"[体检通过] count={d.get('count')},无 std<{STD_FLOOR_CHECK} 的恒定维,json 可直接喂训练")
        return
    print("★[体检报警] 发现恒定维(该维数据全程不动):")
    for s in suspicious:
        print(f"    {s}")
    print(
        "  处理指引:\n"
        "    - 若是有意锁定的自由度(如 v4 固定头):必须打 std 地板,\n"
        "      参照 mjq_lingbotvla_v2/fix_v4_head_std.py(改 NORM_JSON/HEAD_KEYS 后 --apply),\n"
        "      否则 meanstd 归一化 0/0=NaN 直接报废训练;\n"
        "    - 若是意外恒定(采集中某通道没动/没映射错):先查数据,别急着地板。"
    )
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印解析后的参数与体检结果,不真正计算")
    cli_args = ap.parse_args()

    # 切到项目根 + 加入 path(统计脚本依赖相对路径和项目 import)
    os.chdir(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

    # ---- 设分布式环境变量(单卡;arguments.py 的 __post_init__ 强制读这三个) ----
    # WORLD_SIZE=1 → compute_norm_stats 跳过 NCCL 初始化,无需 MASTER_ADDR/PORT
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # 用 0 号卡(5090)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # ---- 离线开关(与训练启动器一致:全本地,不联网) ----
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ---- 设置 sys.argv(compute_norm_stats 用 parse_args 从 sys.argv 读配置) ----
    data_abs = (PROJECT_ROOT / DATA_PATH).resolve()
    norm_abs = (PROJECT_ROOT / NORM_JSON).resolve()
    sys.argv = [
        "compute_norm_stats.py",
        str(PROJECT_ROOT / CONFIG_YAML),
        f"--data.data_name={DATA_NAME}",
        f"--data.train_path={data_abs}",
        f"--data.norm_path={norm_abs}",
    ]

    print("=" * 64)
    print("LingBot-VLA 2.0 norm 统计量计算")
    print(f"  配置:     {CONFIG_YAML}")
    print(f"  数据:     {DATA_PATH}")
    print(f"  产出:     {NORM_JSON}")
    print(f"  单卡:     CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print("=" * 64)

    # ---- 前置检查:路径存在性(算之前就拦住配错) ----
    ok = True
    if not (PROJECT_ROOT / CONFIG_YAML).exists():
        print(f"[错误] 配置不存在: {CONFIG_YAML}"); ok = False
    if not data_abs.exists():
        print(f"[错误] 数据集不存在: {data_abs}"); ok = False
    if not ok:
        sys.exit("[中止] 检查配置区四个路径")
    print(f"[检查] 数据集 OK: {data_abs}")

    if cli_args.dry_run:
        print("[dry-run] 将以下列参数运行(不真算):")
        print(f"  sys.argv = {sys.argv}")
        check_constant_dims(norm_abs)   # 顺手对现有 json 体检(如有)
        return

    # ---- 进程内执行统计脚本(以 __main__ 身份;它会自己打印进度条和耗时) ----
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "compute_norm_stats.py"),
                   run_name="__main__")

    # ---- 算完自动体检(恒定维陷阱) ----
    check_constant_dims(norm_abs)


if __name__ == "__main__":
    main()


# =====================================================================
# 【等价的 CLI 命令】(效果与本脚本相同;多卡时用这种写法)
#   conda activate lingbotv2
#   cd /home/mjq/robot_item/lingbot-vla-v2-main
#   CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py \
#       ./configs/vla/norm_compute/h1_v4_norm.yaml \
#       --data.data_name h1 \
#       --data.train_path /home/mjq/robot_item/lingbot-vla-v2-main/H1_simulation_model/H1_build/data/h1_build_pick_v4_trimmed \
#       --data.norm_path assets/norm_stats/h1_v4.json
#
# 【下一步】若体检报警(恒定维):参照 mjq_lingbotvla_v2/fix_v4_head_std.py 打地板;
#   然后把训练 yaml(Lingbot_H1_Training.py 配置区)指向新数据集 + 新 json。
# =====================================================================
