# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Callable, Dict, List, Literal, Optional

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer, AutoImageProcessor
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import json
from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.dist_utils import main_process_first
from .vla_data import *

logger = logging.get_logger(__name__)

try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    # If datasets or the function doesn't exist, do nothing.
    pass

class DummyDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]

def build_vla_dataset(
    dataset_config,
    model_config,
    config,
    processor,
    do_nomalize = True,
    return_item = False,
    disabled_image_features = False,
    use_depth_align = False,
) -> "Dataset":
    if model_config is None: assert processor is None, "processor and model_config can only be None when computing norm"
    data_name = dataset_config.data_name
    repo_id = dataset_config.train_path
    robot_config_root = dataset_config.robot_config_root
    chunk_size = dataset_config.chunk_size
    prompt_type = dataset_config.prompt_type
    processor = processor if processor is not None and 'qwen' in model_config.tokenizer_path.lower() else None
    img_size = getattr(dataset_config, 'img_size', 256)
    image_augment = bool(getattr(dataset_config, "image_augment", False))
    use_future_image = getattr(dataset_config, 'use_future_image', False)

    if data_name == 'multi':
        dataset = MultiVLADataset(
            repo_file=repo_id,
            dataset_config=dataset_config,
            robot_config_root=robot_config_root,
            config=config,
            processor=processor,
            image_size=(img_size, img_size),
            chunk_size=chunk_size,
            disabled_image_features=disabled_image_features,
            prompt_type=prompt_type,
            do_nomalize=do_nomalize,
            return_item=return_item,
            image_augment=image_augment,
            use_depth_align=use_depth_align,
            use_future_image=use_future_image
        )
    else:
        dataset = VLADataset(
            repo_id=repo_id,
            data_name=data_name,
            robot_config_root=robot_config_root,
            dataset_config=dataset_config,
            config=config,
            processor=processor,
            image_size=(img_size, img_size),
            do_nomalize=do_nomalize,
            return_item=return_item,
            chunk_size=chunk_size,
            disabled_image_features=disabled_image_features,
            image_augment=image_augment,
            use_depth_align=use_depth_align,
            use_future_image=use_future_image
        )

    # 相位加权采样(方案 A,配置门控):data.phase_window_file 非空时把数据集包一层
    # 索引展开映射。默认空字符串 → 原样返回,与历史行为逐比特一致。
    phase_window_file = getattr(dataset_config, "phase_window_file", "")
    if phase_window_file:
        dataset = PhaseWeightedDataset(dataset, phase_window_file)

    return dataset


class PhaseWeightedDataset(Dataset):
    """相位加权采样包装(索引展开式,不改采样器)。

    背景:v4 闭环 0/10 的主根因③是发起段/终段采样不足(40k 步仅 0.35 epoch,
    发起帧平均被访问 0.35 次;见项目根《V4闭环失败诊断.md》)。
    做法:窗口文件里的目标帧在采样空间里复制 weight 份(窗口重叠取最大、不
    叠乘),__len__ 变为展开后长度 —— StatefulDistributedSampler / 断点续训 /
    DataLoader 机制全部不动,均匀 shuffle 自然升级成加权 shuffle。
    窗口文件由 mjq_lingbotvla_v2/make_phase_windows.py 生成(绝对行号)。
    """

    def __init__(self, data: Dataset, window_file: str):
        import numpy as np

        with open(window_file) as f:
            spec = json.load(f)
        n = len(data)
        if int(spec["num_rows"]) != n:
            raise ValueError(
                f"phase_window_file 与数据集不符: json num_rows={spec['num_rows']} "
                f"!= dataset len={n}({window_file})。窗口文件是按别的数据集/副本"
                f"生成的,请对当前 train_path 重跑 make_phase_windows.py。"
            )
        mult = np.ones(n, dtype=np.int64)
        for group in spec["windows"]:
            w = int(group["weight"])
            if w < 1:
                continue
            for s, e in group["ranges"]:
                mult[s:e + 1] = np.maximum(mult[s:e + 1], w)
        self._data = data
        self.sample_map = np.repeat(np.arange(n, dtype=np.int64), mult)

        stats = ", ".join(
            f"w={int(w)}: {int((mult == w).sum())} 帧" for w in np.unique(mult)
        )
        log = getattr(logger, "info_rank0", logger.info)
        log(
            f"PhaseWeightedDataset: {window_file} | {n} -> {len(self.sample_map)} 行"
            f"({len(self.sample_map) / n:.2f}x) | {stats}"
        )

    def __len__(self) -> int:
        return len(self.sample_map)

    def __getitem__(self, index: int):
        return self._data[int(self.sample_map[index])]

class MappingDataset(Dataset):
    """
    Mapping dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]


class IterativeDataset(IterableDataset):
    """
    Iterative dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            if self._transform is not None:
                yield self._transform(sample)
            else:
                yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        self._data.set_epoch(epoch)


def build_dummy_dataset(size: int, max_seq_len: int) -> "Dataset":
    return DummyDataset(size=size, seq_length=max_seq_len)


def build_mapping_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
) -> "Dataset":
    """
    Build mapping dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
    Returns:
        Dataset: mapping dataset
    """
    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_path)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    with main_process_first():
        dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace)

    return MappingDataset(data=dataset, transform=transform)


def build_iterative_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
) -> "IterableDataset":
    """ "
    Build iterative dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        seed (int): random seed
    Returns:
        IterableDataset: iterative dataset
    """

    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_path)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    parallel_state = get_parallel_state()
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    return IterativeDataset(dataset, transform)
