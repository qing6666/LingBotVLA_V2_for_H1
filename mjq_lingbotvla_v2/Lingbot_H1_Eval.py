#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lingbot_H1_Eval.py —— H1 开环评估(open-loop eval)纯 Python 启动器(多 checkpoint 对比版)
====================================================================
【这个脚本做什么】
  拿训练产出的 checkpoint,在验证数据上做"开环评估":
  给模型看真实的(图像+状态+指令),把它预测的动作和 数据集里人类/仿真记录的真实动作
  逐帧对比,算 MSE/MAE —— 动作误差越小,模型学得越好。
  ★ 与训练 loss 的区别:loss 是训练集拟合度(训练时见过);开环评估是独立衡量
    "动作预测准不准",挑最佳 checkpoint 以此为准(晚训的档不一定更好,可能过拟合)。

【怎么调用的】★ 纯 Python 进程内执行 scripts/open_loop_eval.py
  和 Lingbot_H1_Training.py 同一套路,但有一点不同:
  open_loop_eval.py 的启动逻辑全在 `if __name__ == "__main__":` 里(argparse 驱动,
  没有无参 main() 可 import)→ 本脚本用 runpy.run_path(..., run_name="__main__")
  以受控的 sys.argv 执行它 —— 与 CLI 命令行行为 100% 等价。
  循环多档 checkpoint,每档跑完释放显存(gc + empty_cache)再载下一档,
  最后汇总对比表 + 推荐最佳档。

【前置条件】
  1. conda activate lingbotv2(或让脚本自纠正)
  2. 训练已完成:output/h1/checkpoints/global_step_*/hf_ckpt 存在
  3. ★ 不要移动 hf_ckpt 的位置 —— 评估的 policy 自动识别靠
     output/h1/lingbotvla_cli.yaml(它在 hf_ckpt 往上三级的位置,官方写死的查找规则)

【产出】
  终端:每档的逐轨迹 MSE/MAE + 三档对比表
  图片:SAVE_ROOT/<step>/<traj_id>.png(每条轨迹的预测-vs-真值动作曲线)

【怎么跑】
  ★ VSCode:打开本文件点右上角 ▶
  ★ 终端:python mjq_lingbotvla_v2/Lingbot_H1_Eval.py
  ★ 只评某一档:python mjq_lingbotvla_v2/Lingbot_H1_Eval.py 20000
    (命令行里给档位数字,覆盖下方 CHECKPOINT_STEPS)
====================================================================
"""

import os
import sys
import gc
import re
import runpy
from contextlib import redirect_stdout
from pathlib import Path

# ============ 解释器自纠正保险(同训练启动器) ============
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}")
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# =======================================================

# ==================== 配置(改成你的参数)====================
# ★ 当前评估"裁剪数据重训"的产物(h1_trim:98 episodes,loader 修复 + 静止前缀已裁)。
#   切回评估旧训练时改回:
#     CKPT_ROOT/SAVE_ROOT 的 output/h1_trim → output/h1
#     DATA_PATH = ".../h1_build_pick_v2";NORM_STATS = "assets/norm_stats/h1.json"
#     TRAJ_IDS 里的 97 → 100(旧数据 101 条,编号 0~100)
CHECKPOINT_STEPS = ["2500", "5000", "7500", "10000", "12500", "15000", "17500", "20000"]
                                                # 全部 8 档对比(传命令行参数可覆盖)
CKPT_ROOT = "output/h1_trim/checkpoints"        # 训练产出的 checkpoint 根目录
DATA_PATH = "H1_simulation_model/H1_build/data/h1_build_pick_v2_trimmed"  # ★与训练同一份(裁剪副本)
ROBO_NAME = "h1"                                 # robot_config 名(与训练一致)
NORM_STATS = "assets/norm_stats/h1_trim.json"    # ★必须与训练时同一份(去归一化要还原到同一尺度)
TRAJ_IDS = [0, 10, 20, 30, 50, 70, 90, 97]       # 抽 8 条轨迹评估(裁剪副本共 98 条,编号 0~97;
                                                #   ★ 97 是最后一条;旧数据集的 100 已不存在)
USE_LENGTH = 50                                  # 动作块长度(=训练 chunk_size,别改)
MAX_INFER_TIME = 10                              # 每条轨迹最多推理次数(默认 10)
USE_BF16 = True                                  # bf16 推理(与训练同精度,显存 12GB);
                                                #   False=fp32 评估(24GB,数值更精细)
SAVE_ROOT = "output/h1_trim/open_loop_eval"      # 每档曲线图的输出根目录
# =============================================================

# 项目根目录(本脚本在 mjq_lingbotvla_v2/ 下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "open_loop_eval.py"


class _Tee:
    """stdout 分流器:终端实时看 + 存进缓冲供事后解析(不像 redirect 会吞掉输出)。"""

    def __init__(self):
        self.buffer = []

    def write(self, s):
        sys.__stdout__.write(s)
        self.buffer.append(s)
        return len(s)

    def flush(self):
        sys.__stdout__.flush()

    @property
    def text(self):
        return "".join(self.buffer)


def evaluate_one_checkpoint(step: str):
    """跑一档 checkpoint,返回 (avg_mse, avg_mae, 轨迹数)。"""
    hf_ckpt = PROJECT_ROOT / CKPT_ROOT / f"global_step_{step}" / "hf_ckpt"
    if not hf_ckpt.exists():
        print(f"[跳过] {hf_ckpt} 不存在")
        return None

    # ---- 构造与 CLI 完全等价的 sys.argv(open_loop_eval.py 用 argparse 读它)----
    sys.argv = [
        "open_loop_eval.py",
        f"--model_path={hf_ckpt}",
        f"--robo_name={ROBO_NAME}",
        f"--norm_path={NORM_STATS}",
        f"--data_path={(PROJECT_ROOT / DATA_PATH).resolve()}",
        "--traj_ids", *[str(t) for t in TRAJ_IDS],
        f"--use_length={USE_LENGTH}",
        f"--max_infer_time={MAX_INFER_TIME}",
        f"--save_plot_path={(PROJECT_ROOT / SAVE_ROOT / f'global_step_{step}').resolve()}",
    ] + (["--use_bf16"] if USE_BF16 else [])

    tee = _Tee()
    try:
        with redirect_stdout(tee):
            runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")
    finally:
        # 模型已无引用,主动回收:Python 循环引用 + CUDA 缓存池,给下一档腾显存
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 从输出里解析汇总指标 ----
    text = tee.text
    m_mse = re.search(r"Average MSE across all trajs:\s*([0-9.eE+-]+)", text)
    m_mae = re.search(r"Average MAE across all trajs:\s*([0-9.eE+-]+)", text)
    n_traj = len(re.findall(r"MSE for trajectory", text))
    if not (m_mse and m_mae):
        print(f"[警告] 档位 {step} 没解析到汇总指标,检查上方日志")
        return None
    return float(m_mse.group(1)), float(m_mae.group(1)), n_traj


def main():
    os.chdir(str(PROJECT_ROOT))                 # 相对路径(configs/、assets/)都以项目根为基准
    sys.path.insert(0, str(PROJECT_ROOT))       # ★让 lingbotvla 包解析到本仓库(优先于任何
                                                #   site-packages 里可能存在的其他安装)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"          # 全本地权重,不联网
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    steps = sys.argv[1:] or CHECKPOINT_STEPS    # 命令行给了档位就用命令行的

    print("=" * 64)
    print("LingBot-VLA 2.0 开环评估(多 checkpoint 对比)")
    print(f"  档位:   {', '.join(f'step_{s}' for s in steps)}")
    print(f"  数据:   {DATA_PATH}")
    print(f"  轨迹:   {TRAJ_IDS}")
    print(f"  精度:   {'bf16' if USE_BF16 else 'fp32'}")
    print(f"  曲线图: {SAVE_ROOT}/<step>/<traj_id>.png")
    print("=" * 64)

    results = {}                                # step -> (mse, mae, n_traj)
    for i, step in enumerate(steps, 1):
        print(f"\n{'='*24} [{i}/{len(steps)}] global_step_{step} {'='*24}")
        r = evaluate_one_checkpoint(step)
        if r:
            results[step] = r
            print(f"\n>>> 档位 {step} 汇总: MSE={r[0]:.6f}  MAE={r[1]:.6f}  (轨迹数 {r[2]})")

    # ---------- 最终对比表 ----------
    print("\n" + "=" * 64)
    print("开环评估对比结果(按 MSE 升序,MSE/MAE 越小越好)")
    print("=" * 64)
    if not results:
        print("没有成功的档位,检查上方日志。")
        return
    rows = sorted(results.items(), key=lambda kv: kv[1][0])
    print(f"{'checkpoint':<22}{'平均MSE':>14}{'平均MAE':>14}{'轨迹数':>8}")
    for step, (mse, mae, n) in rows:
        print(f"{'global_step_' + step:<22}{mse:>14.6f}{mae:>14.6f}{n:>8d}")
    best = rows[0][0]
    print(f"\n推荐最佳档: global_step_{best}(动作误差最小)")
    print(f"部署/仿真用权重: {CKPT_ROOT}/global_step_{best}/hf_ckpt")
    print(f"逐轨迹预测-vs-真值曲线: {SAVE_ROOT}/global_step_{best}/<traj_id>.png")


if __name__ == "__main__":
    main()


# =====================================================================
# 【等价的 CLI 命令】(单档评估,效果与本脚本对单档的处理完全相同)
#   conda activate lingbotv2
#   cd /home/mjq/robot_item/lingbot-vla-v2-main
#   python scripts/open_loop_eval.py \
#       --model_path output/h1/checkpoints/global_step_20000/hf_ckpt \
#       --robo_name h1 \
#       --norm_path assets/norm_stats/h1.json \
#       --data_path /home/mjq/robot_item/lingbot-vla-v2-main/H1_simulation_model/H1_build/data/h1_build_pick_v2 \
#       --traj_ids 0 10 20 30 50 70 90 100 \
#       --use_length 50 \
#       --max_infer_time 10 \
#       --use_bf16 \
#       --save_plot_path output/h1/open_loop_eval/global_step_20000
#
# 【常用变体】
#   # 换一档评估(把 20000 换成 2500/5000/7500/.../17500 均可):
#       ... --model_path output/h1/checkpoints/global_step_15000/hf_ckpt ...
#   # fp32 评估(更精细,显存 24GB):去掉 --use_bf16
#   # 单条轨迹快速冒烟测试:--traj_ids 0
#
# 【注意】
#   * --policy 默认 auto:靠 output/h1/lingbotvla_cli.yaml 自动选 policy 实现,
#     该文件在 hf_ckpt 往上三级处 —— 别移动 hf_ckpt,否则 auto 失效需手动指定
#   * --norm_path 必须与训练时同一份统计表,否则去归一化尺度错位,误差数字无意义
# =====================================================================
