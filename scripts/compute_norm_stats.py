# -*- coding: utf-8 -*-
"""
compute_norm_stats.py —— 归一化统计量计算脚本
"流式统计器 + 多进程加速 + 多卡合并",把整库数据压成一个小 JSON
====================================================================
【作用】扫描整个 LeRobot 数据集,为每个统一状态/动作特征(如 arm.position、
        effector.position)在线累计统计量(mean / std / min / max / q01 / q99 ...),
        最终保存成一个 JSON(norm_stats.json)。训练时 VLADataset 读取该 JSON,
        把原始数据归一化到适合模型学习的范围。

【为什么需要】不同关节量纲差异大(角度/位置/夹爪开合),VLA 模型对数值范围敏感,
             必须先统计、再归一化。

【为什么"在线"】数据集可能几十万条,无法一次性载入内存,所以用流式统计(RunningStats)
              逐 batch 更新;并支持多卡分布式(每卡算一份,最后 merge)。

【典型运行】
    bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
        --data.norm_path assets/norm_stats/robotwin.json
====================================================================
"""

import json
import numpy as np
import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import trange, tqdm
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch.multiprocessing as mp
import torch
import torch.distributed as dist

from lingbotvla.data import build_vla_dataset
from lingbotvla.utils.normalize import (
    RunningStats,
    RunningStatsState,
)
from lingbotvla.models import build_processor
from lingbotvla.utils import helper
from lingbotvla.utils.arguments import parse_args
from lingbotvla.utils.dist_utils import all_reduce
import lingbotvla.utils.normalize as normalize

# 把项目根目录加入 sys.path,以便 import tasks.vla.train_lingbotvla 里的参数类
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 复用训练脚本里定义的参数类,保证"算统计量"与"训练"使用完全相同的 data/train 参数定义
from tasks.vla.train_lingbotvla import MyTrainingArguments, MyDataArguments

logger = helper.create_logger(__name__)

@dataclass
class NormComputeDataArguments(MyDataArguments):
    """归一化计算专用的数据参数(继承训练用的 MyDataArguments,新增 4 个字段)。"""

    data_ratio_for_norm_compute: float = field(
        default=1.0,
        metadata={"help": "采样比例。<1.0 时只随机抽取该比例的数据来算统计量(用于加速)。"},
    )
    robot_name: str = field(
        default=None,
        metadata={"help": "只对指定的 robot config 名称计算统计量(逗号分隔多个);None 表示全部。"},
    )
    norm_path: str = field(
        default=None,
        metadata={"help": "统计量 JSON 的保存路径。"},
    )
    norm_merge_chunk_dim: bool = field(
        default=True,
        metadata={"help": "计算 action 统计量时,是否把 chunk(动作序列)维度合并进特征维度一起统计。"},
    )


@dataclass
class Arguments:
    """顶层参数容器:分 data 与 train 两块。"""
    data: "NormComputeDataArguments" = field(default_factory=NormComputeDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def get_all_tasks(task_files, robot_name, sep=' '):
    """解析多数据集列表文件(每行 '<robot_config_name> <数据集路径>')。

    Args:
        task_files: 逗号分隔的多个 .txt 路径。
        robot_name: 只保留这些 robot config 名称(过滤);None 表示全保留。
        sep: 行内两列的分隔符,默认空格。
    Returns:
        data_names: 每个 task 对应的 robot config 名列表。
        task_list:  每个 task 对应的数据集路径列表。
    """
    task_files = task_files.split(',')
    data_names, task_list = [], []
    for task_file in task_files:
        assert task_file.lower().endswith('.txt')
        with open(task_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_name, task = line.split(sep)
                if robot_name is not None and data_name not in robot_name:
                    continue
                data_names.append(data_name)
                task_list.append(task)
        f.close()
    return data_names, task_list


def collate_dict(batch_list):
    """
    把一个 batch 的样本列表拼成单个大 dict(简易 collate 函数)。
    将 [{ 'a': t1, 'b': t2 }, { 'a': t3, 'b': t4 }]
    转成 { 'a': tensor([t1, t3]), 'b': tensor([t2, t4]) }
    只对 Tensor 字段做 stack,其余字段忽略(统计只需要数值张量)。
    """
    keys = batch_list[0].keys()
    batch = {}
    for key in keys:
        # If it is a Tensor, stack them together
        if isinstance(batch_list[0][key], torch.Tensor):
            batch[key] = torch.stack([item[key] for item in batch_list])
    return batch

# ============== 多进程数据加载优化 ==============
# 思路:用 Pool 的 initializer 让每个 worker 进程只初始化一次 dataset,
#       之后只传 indices 给 worker,避免每个 task 都把整个 dataset 序列化过去。
_global_dataset = None

def init_worker(dataset):
    """worker 进程启动时调用一次:把 dataset 绑到全局变量,后续复用。"""
    global _global_dataset
    _global_dataset = dataset

def worker_fn(indices):
    """worker 主逻辑:按 indices 取一批样本并 collate 成 batch dict。"""
    global _global_dataset
    # 1. Fetch each dict one by one
    samples = [_global_dataset[i] for i in indices]
    # 2. Collate the list of dicts into a single large dict (Batch)
    batch = collate_dict(samples)
    return batch

def get_batch_indices(target_ids, batch_size):
    """把所有样本 id 切成若干个 batch(每个含 batch_size 个 id)。"""
    return [target_ids[i:i + batch_size] for i in range(0, len(target_ids), batch_size)]


def compute_norm(dataset, batch_size, stats, state_norm_keys, acton_norm_keys, delta_norm, ratio,
                 rank=0, world_size=1, num_workers=8, norm_merge_chunk_dim=False):
    """核心函数:多进程遍历数据集,逐 batch 在线更新各特征的 RunningStats。

    Args:
        dataset: MultiVLADataset(已关闭图像加载与归一化)。
        batch_size: 每个 batch 的样本数(取自 train.micro_batch_size)。
        stats: {feature_key: RunningStats};会被原地更新。
        state_norm_keys / acton_norm_keys: 要统计的状态/动作特征名列表。
        delta_norm: {feature_key: bool},该动作特征是否为"相对动作"(subtract_state=True)。
        ratio: 采样比例(<1 只用部分数据)。
        rank / world_size: 分布式分片参数。
        num_workers: 多进程 worker 数。
        norm_merge_chunk_dim: action 是否把 chunk 维合并进特征维一起统计。
    """
    if ratio < 1:
        num_step = int(len(dataset)*ratio)
        # 固定随机种子,保证每个 rank 采到同一份"全局子集",再各自跨步切片得到不相交的分片
        random.seed(42)
        data_ids = random.sample(range(len(dataset)), num_step)
    else:
        data_ids = list(range(len(dataset)))   # ratio==1:用全部样本

    # 按 rank 跨步切片:rank 处理 data_ids[rank], data_ids[rank+world_size], ...
    # 用"跨步"而非"连续切块",可缓解各子数据集大小不均导致的负载不均衡
    data_ids = data_ids[rank::world_size]

    mp.set_start_method('fork', force=True)

    all_batch_indices = get_batch_indices(data_ids, batch_size)
    total_batches = len(all_batch_indices) # Total number of batches

    # 进程池:initializer 让每个 worker 只建一次 dataset;
    # imap_unordered 谁先完成谁先返回(不保序,效率最高),现在只传 indices,不传整个 dataset
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(dataset,)) as pool:
        # imap_unordered returns results as soon as they are ready without preserving
        # order, which is the most efficient. Only indices are passed now, not the
        # whole dataset.
        results_generator = pool.imap_unordered(worker_fn, all_batch_indices)

        pbar = tqdm(
            results_generator,
            total=total_batches,
            unit="batch",
            ncols=100,
            disable=(rank != 0),     # 只在 rank0 显示进度条
            desc=f"rank{rank}",
        )
        for batch in pbar:
            # ---- 用本 batch 更新各特征的在线统计量 ----
            for key in state_norm_keys:
                values = np.asarray(batch[key])
                # reshape 成 (N, feature_dim):把前面所有维度拍平,只保留最后一维(特征维)
                stats[key].update(values.reshape(-1, values.shape[-1]))
            for key in acton_norm_keys:
                # 若该动作是"相对动作"且不合并 chunk 维:把 chunk 维独立(每个时间步当不同特征统计)
                values = np.asarray(batch[key]) if (not delta_norm[key] or norm_merge_chunk_dim) else np.asarray(batch[key].reshape(batch[key].shape[0], -1))
                stats[key].update(values.reshape(-1, values.shape[-1]))

    del pool
    del dataset

def get_norm_stats(stats, delta_norm, chunk_size, norm_merge_chunk_dim=False):
    """从累计好的 RunningStats 导出最终统计量(mean/std/q01/q99/min/max...)。

    对"相对动作"且不合并 chunk 维的特征,按 chunk_size 把统计量 reshape 回 (chunk, feat/chunk)。
    """
    assert stats is not None
    norm_stats = {}
    for key, state in stats.items():
        _chunk_size = chunk_size if (key in delta_norm and delta_norm[key]==True) and not norm_merge_chunk_dim else None
        norm_stats[key] = state.get_statistics(chunk_size=_chunk_size)
    return norm_stats


def _init_dataset_worker(
    args, data_names
) -> 'LeRobotDataset':
    """构造数据集。两个关键开关:
       - do_nomalize=False:        不做归一化(要用原始值算统计量,否则就是拿归一化值算归一化参数了)
       - disabled_image_features=True: 不加载图像(算统计量只需要 state/action 数值,省时省内存)
    """
    args.data.chunk_size = args.train.chunk_size
    dataset = build_vla_dataset(dataset_config=args.data,
                                model_config=None,
                                config=None,
                                processor=None,
                                do_nomalize = False,
                                return_item = True,
                                disabled_image_features = True)

    return dataset

if __name__ == "__main__":

    # 1) 解析参数(命令行 + yaml,复用训练参数体系)
    args = parse_args(Arguments)

    # Distributed initialization: reuse train.sh + torchrun; RANK/WORLD_SIZE/LOCAL_RANK are already injected via env vars
    # 2) 分布式初始化:复用 train.sh + torchrun,rank/world_size/local_rank 由环境变量注入
    if args.train.world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(f"cuda:{args.train.local_rank}")
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    rank = args.train.global_rank
    world_size = args.train.world_size

    logger.info(f"Process rank: {rank}, world size: {world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    logger.info_rank0("Prepare data")
    stats = None

    # 本脚本只支持 VLA 类型数据
    assert args.data.datasets_type == 'vla'

    # robot_name 支持逗号分隔多个(为 None 表示不过滤)
    robot_name = args.data.robot_name.split(',') if args.data.robot_name is not None else None

    # 3) 解析数据集清单:单数据集 or multi(读 .txt 列表)
    if args.data.data_name == 'multi':
        data_names, repo_ids = get_all_tasks(args.data.train_path, robot_name)
        if robot_name is None:
            # 没指定 robot_name 时,要求列表里所有行的 robot config 名一致
            assert len(set(data_names)) == 1
        else:
            for data_name in set(data_names):
                assert data_name in robot_name
    else:
        # 单数据集:也统一转成 multi 形式(1 行),复用同一套多数据集加载逻辑
        data_names, repo_ids = [args.data.data_name], [args.data.train_path]
        args.data.data_name = 'multi'


    # 4) 把数据集清单写成临时 txt,再交给 build_vla_dataset 以 multi 模式加载
    filename = '_'.join(list(set(data_names)))
    tmp_dir = f"tmp/"
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    filename = os.path.join(tmp_dir, f"tmp_{filename}_rank{rank}.txt")
    with open(filename, 'w') as f:
        for robot, task in zip(data_names, repo_ids):
            f.write(f"{robot} {task}\n")
    f.close()
    args.data.train_path = filename
    dataset = _init_dataset_worker(args, data_names)
    if rank == 0:
        print(f"===========\nProcessing {len(dataset._datasets)} lerobot datasets\n===========")
    os.remove(filename)   # 临时文件用完即删
    # 断言:所有子数据集的"状态特征+动作特征"命名必须完全一致(否则无法共用一套统计量)
    assert len(list(set([' '.join(_datasets.state_features+_datasets.action_features) for _datasets in dataset._datasets])))==1

    # 5) 取出要统计的特征名 + 每个 RunningStats 实例
    state_norm_keys = dataset._datasets[0].state_features
    acton_norm_keys = dataset._datasets[0].action_features
    delta_norm = dataset._datasets[0].feature_transform.action_subtract_state   # 各动作是否为相对量
    stats = {key: normalize.RunningStats() for key in acton_norm_keys+state_norm_keys}  # 每个特征一个累计器
    chunk_size = args.data.chunk_size

    # 6) 多进程流式扫描数据集,累计统计量
    ratio = args.data.data_ratio_for_norm_compute
    compute_norm(dataset, args.train.micro_batch_size, stats, state_norm_keys, acton_norm_keys,
                 delta_norm, ratio=ratio, rank=rank, world_size=world_size,
                 num_workers=args.data.num_workers, norm_merge_chunk_dim=args.data.norm_merge_chunk_dim)

    # 7) 跨 rank 合并:每个 rank 把本地 stats 序列化后 all_gather_object 给所有 rank;
    #    rank0 执行真正的 merge 并落盘
    if world_size > 1:
        local_state = {
            k: (v.get_state().model_dump() if v._count > 0 else None)
            for k, v in stats.items()
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_state)
        if rank == 0:
            merged = {}
            for key in stats.keys():
                objs = []
                for shard in gathered:
                    if shard is None or shard.get(key) is None:
                        continue
                    objs.append(RunningStats.from_state(RunningStatsState(**shard[key])))
                if not objs:
                    raise RuntimeError(f"No rank produced any data for key={key!r}")
                merged[key] = RunningStats.merge(objs)   # 合并多卡的直方图/均值/极值
            stats = merged
        dist.barrier()

    # 8) rank0 导出最终统计量并保存为 JSON
    if rank == 0:
        norm_stats = get_norm_stats(stats, delta_norm, chunk_size, args.data.norm_merge_chunk_dim)
        output_path = Path(args.data.norm_path)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats, stats[state_norm_keys[0]]._count)

    if world_size > 1:
        dist.destroy_process_group()
