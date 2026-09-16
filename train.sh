#!/bin/bash
# =====================================================================
# train.sh —— LingBot-VLA 2.0 的"通用分布式启动器"
# ---------------------------------------------------------------------
# 它本身不关心要跑什么任务,只做三件事:
#   ① 设置运行环境变量(离线模式、tokenizer 并行开关等)
#   ② 自动探测本机 GPU 数量
#   ③ 用 torchrun 拉起分布式进程,把用户传入的脚本+配置透传过去
#
# 因此它是个"万能启动器":训练、算归一化统计、评估都能用它启动。
# 用法:
#   bash train.sh <python脚本> <配置.yaml> [--参数 值 ...]
# 例:
#   bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml
#   bash train.sh scripts/compute_norm_stats.py   configs/vla/robotwin/robotwin.yaml
# 全部输出同时写入 log.txt。
# =====================================================================

set -x   # 执行前先打印每条命令(便于调试,看实际跑了什么)

# ---- ① 环境变量 ----
export TOKENIZERS_PARALLELISM=false    # 关闭 HuggingFace tokenizer 多进程并行,避免和 DataLoader 的 worker 抢锁死锁
export HF_HUB_OFFLINE=1                # 强制离线:不去 HF Hub 拉模型/数据(用本地缓存)
export HF_DATASETS_OFFLINE=1           # 强制离线:datasets 库不联网
export TRANSFORMERS_OFFLINE=1          # 强制离线:transformers 库不联网
export HF_HUB_DISABLE_TELEMETRY=1      # 关闭 HF 遥测上报
export DISABLE_TELEMETRY=1             # 关闭通用遥测上报

# ---- ② 自动探测本机 GPU 数量 ----
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  # 没指定 CUDA_VISIBLE_DEVICES:用 nvidia-smi 数物理 GPU 数
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  # 指定了 CUDA_VISIBLE_DEVICES(如 "0,1,2"):数逗号个数
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"

# ---- ③ 分布式参数(都是"有则用环境变量的值,无则用默认值") ----
# 语法 ${VAR:=默认值}:VAR 未设/为空时赋为默认值并使用,已设则沿用。便于多机时用环境变量覆盖。
NNODES=${NNODES:=1}                # 参与训练的机器(节点)数,默认单机=1
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}  # 每台机器的 GPU 数(用上面探测到的)
NODE_RANK=${NODE_RANK:=0}          # 当前机器的编号(多机时 0,1,2...),默认 0
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}   # 主节点 IP(多机通信用),默认本机
MASTER_PORT=${MASTER_PORT:=62500}  # 主节点通信端口,默认 62500


# ---- 用 torchrun 拉起分布式进程 ----
# torchrun 会启动 NNODES × NPROC_PER_NODE 个进程,并给每个进程注入:
#   RANK(全局序号)、WORLD_SIZE(总进程数)、LOCAL_RANK(机内 GPU 序号)
# 这些环境变量会被 Python 脚本(train_lingbotvla.py / compute_norm_stats.py)读取,
# 用来 init_process_group 初始化 NCCL 分布式。
#
# 末尾的 $@ 把命令行里 train.sh 之后的所有参数(脚本路径+配置+--xxx)原样透传给 torchrun。
# 2>&1 | tee log.txt:把 stdout+stderr 同时显示到屏幕并写入 log.txt。
# 注意:下面第一行末尾的 '\' 是续行符,其后不能有任何字符(包括注释)。
torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
