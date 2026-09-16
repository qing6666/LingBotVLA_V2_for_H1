# Copyright 2026 Robbyant Team and/or its affiliates
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

# -*- coding: utf-8 -*-
"""
base_dataset.py —— VLA 单数据集加载核心
====================================================================
本文件是训练时数据流的"取数 + 组织"层,包含两个类:

  1) LeRobotDataset :继承 LeRobot 官方数据集,扩展了"按 delta_indices 取未来
                      序列(action chunk)"、"按时间戳抽视频帧"、"load_image 开关"等。
                      负责:从 parquet/视频文件里把一条样本(含未来动作序列)取出来。

  2) VLADataset     :VLA 训练用的 Dataset 包装器。负责:加载 robot config、构造
                      FeatureTransform(字段映射+归一化)、配置 action chunk 的时间窗、
                      把底层取出的原始样本交给 FeatureTransform 转成模型输入。

  数据流:DataLoader → VLADataset.__getitem__ → getitem → LeRobotDataset.__getitem__(取原始)
                       → FeatureTransform.apply(映射 + 归一化 + padding) → 模型输入 dict
====================================================================
"""


import os
import inspect
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize

# 兼容 LeRobot v3(新路径)与 v2(旧路径)两套 API
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import hf_transform_to_torch
    LEROBOT_DATASET_API = "v3"
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import hf_transform_to_torch
    LEROBOT_DATASET_API = "v2"

from datasets import load_dataset as _hf_load_dataset

from ...utils import logging
from .utils import FeatureTransform
from .video_utils import decode_video_frames


logger = logging.get_logger(__name__)


def _to_relative_indices(dataset, query_indices):
    """把绝对样本索引转成"episode 内相对索引"(LeRobot v3 多 episode 连续存储时需要)。"""
    index_map = getattr(dataset, "_absolute_to_relative_idx", None)
    if index_map is None:
        return query_indices
    return [index_map[idx] for idx in query_indices]


def _get_task_name(tasks, task_idx):
    """按 task_index 取任务名(兼容 pandas DataFrame 与普通 list 两种 tasks 结构)。"""
    if hasattr(tasks, "iloc"):
        return tasks.iloc[task_idx].name
    return tasks[task_idx]


def _resolve_lerobot_location(repo_id):
    """判断 repo_id 是"本地目录"还是"HuggingFace repo id"。
    返回 (repo_id 用于 metadata, root 本地路径 or None)。"""
    repo_path = Path(repo_id).expanduser()
    if repo_path.exists():
        return repo_path.name, repo_path
    return repo_id, None


def _filter_supported_kwargs(callable_obj, kwargs):
    """只保留目标 callable 实际支持的参数(用于跨 LeRobot 版本的签名差异兼容)。"""
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}



class LeRobotDataset(BaseLeRobotDataset):
    """对官方 LeRobotDataset 的轻量扩展:加了 load_image 开关、批量按索引查询、视频抽帧。"""

    def __init__(
        self,
        repo_id: str,
        load_image: bool = True,
        **kwargs,
    ):
        super().__init__(repo_id, **kwargs)
        self.load_image = load_image   # 是否加载图像(算 norm 统计量时设 False 以省时省内存)

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.
        按"键→索引列表"批量从 parquet 取数(跳过 video key,视频另有 _query_videos 处理)。

        Tries column-first [key][indices] for speed, falls back to row-first.
        列优先(更快),失败则退回逐行。

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue   # 视频键跳过(由 _query_videos 按时间戳抽帧)
            # Map absolute indices to relative indices if needed
            relative_indices = _to_relative_indices(self, q_idx)
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])   # 取出并堆叠成序列
        return result

    def load_hf_dataset(self, features=None):
        """把所有 episode 的 parquet 文件加载成一个 HF Dataset(可只选需要的列以省内存)。"""
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # ★ 去重保序:v3 里多个 episode 共享同一个 parquet 分片文件,上面按 episode
        #   展开的文件列表必然含重复;datasets.load_dataset 不去重 → hf 行数被放大成
        #   "副本数×每文件行数"(h1 数据实测 78108 → 3935163 行),位置索引与全局
        #   index/episode 边界全部错位:idx≥首文件行数的样本读到错位行,
        #   action chunk 被错误 clamp 成"末帧动作×50"(v2 损失不用 action_is_pad,
        #   不掩码)→ 半数批次是错配监督。dict.fromkeys 保序去重后位置==index。
        files = list(dict.fromkeys(files))
        hf_dataset = _hf_load_dataset("parquet", data_files=files, split="train")
        if features is not None:
            # available = set(hf_dataset.column_names)
            # features = [f for f in features if f in available]
            hf_dataset = hf_dataset.select_columns(features)   # 只保留需要的列

        hf_dataset.set_transform(hf_transform_to_torch)   # 设置取出时自动转 torch 张量
        return hf_dataset

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """按时间戳从 mp4 抽取视频帧(图像)。
        Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        警告:使用多 worker 的 DataLoader 时,不要在主进程调用本函数(会导致段错误)。
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            if LEROBOT_DATASET_API == "v3":
                # LeRobot v3 stores episodes sequentially in a shared mp4, so
                # query timestamps are relative to the episode start.
                # v3 把多个 episode 连续存进同一个 mp4,查询时间戳要加上该 episode 的起始偏移
                ep = self.meta.episodes[ep_idx]
                from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
                query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)   # 去掉多余的 batch 维

        return item

    def __getitem__(self, idx) -> dict:
        """底层取数:取第 idx 条样本,并按 delta_indices 取出"未来序列(action chunk)"与视频帧。
        """
        # Ensure dataset is loaded when we actually need to read from it
        item = self.hf_dataset[idx]                       # 当前帧的全部字段
        ep_idx = item["episode_index"].item()            # 所属 episode

        query_indices = None
        if self.delta_indices is not None:
            # 按 delta_indices 批量取未来若干帧的 state/action(即 action chunk 的来源)
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}                   # 合并 padding 信息(episode 边界处补齐)
            for key, val in query_result.items():
                item[key] = val                          # 用序列值覆盖单帧值

        if len(self.meta.video_keys) > 0 and self.load_image:
            # 取视频帧(当前帧 + 可能的未来帧)
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None and self.load_image:
            # 图像尺寸变换(如 Resize 到 224×224)
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = _get_task_name(self.meta.tasks, task_idx)   # 把任务描述文本加进去(作为语言指令)

        return item

class VLADataset(Dataset):
    """VLA 训练数据集。包装 LeRobotDataset,叠加"字段映射 + 归一化 + action chunk"逻辑。"""

    def __init__(
        self,
        repo_id,                       # LeRobot 数据集 repo id 或本地路径
        data_name,                     # robot config 名(决定读 configs/robot_configs/<data_name>.yaml)
        dataset_config,                # 数据参数(data.joints / cameras / norm_type 等)
        robot_config_root,             # robot config 所在目录
        config=None,                   # 模型 config(归一化时需要,判断动作维度等)
        processor=None,                # 视觉/文本 processor(图像预处理、tokenizer)
        video_backend = 'torchcodec',  # 视频解码后端
        chunk_size = 50,               # 动作序列长度(模型一次预测多少步未来动作)
        image_size = (224, 224),       # 图像统一尺寸
        do_nomalize = True,            # 是否做字段映射+归一化(算 norm 统计时设 False)
        return_item = False,           # 是否在 padding 前返回(影响返回的 action 是否含 padding)
        disabled_image_features = False,  # 是否禁用图像特征(算 norm 时设 True)
        feature_transform = None,      # 可外部传入已构造的 FeatureTransform(多数据集共享时用)
        use_subtask_as_prompt = False,
        transform=None,                # 额外的样本变换
        image_augment = False,         # 训练时图像增强
        use_depth_align = False,       # 是否启用 depth 对齐(native-depth 训练)
        use_future_image = False,      # 是否加载未来帧(depth/video 蒸馏用)
    ):
        if do_nomalize and config is None:
            raise ValueError("VLADataset requires a model config; pass model.config via build_vla_dataset.")

        self.processor = processor
        self.config = config
        self.chunk_size = chunk_size
        self.data_name = data_name
        self.disabled_image_features = disabled_image_features
        self.use_depth_align = use_depth_align
        self.use_future_image = use_future_image

        # 只在做归一化训练时才需要真正加载图像;算 norm 统计时关掉以省时省内存
        load_image = True if do_nomalize else False

        if feature_transform is None:
            # 构造字段映射器+归一化器:解析 robot config,建立"原始字段 → 统一特征"的映射,
            # 并加载 norm_stats.json 构造 Normalizer
            robot_config = os.path.join(robot_config_root, f'{data_name}.yaml')
            self.feature_transform = FeatureTransform(robot_config, dataset_config, self.config, \
                        processor, disabled_image_features, do_nomalize, \
                        chunk_size=chunk_size, return_item_befor_padding=return_item,\
                        image_augment=image_augment, use_depth_align=use_depth_align,
                        use_future_image=use_future_image)
        else:
            self.feature_transform = feature_transform

        # 暴露统一的特征名列表(供 collator / 模型 / norm 统计使用)
        self.action_features = self.feature_transform.actions
        self.state_features = self.feature_transform.states
        self.image_features = self.feature_transform.images

        # 解析数据集位置(本地 or HF),读 metadata(含 fps、总帧数、视频键等)
        lerobot_repo_id, lerobot_root = _resolve_lerobot_location(repo_id)
        metadata_kwargs = _filter_supported_kwargs(
            LeRobotDatasetMetadata.__init__,
            {"repo_id": lerobot_repo_id, "root": lerobot_root},
        )
        self.dataset_meta = LeRobotDatasetMetadata(**metadata_kwargs)
        # 合并"动作/状态序列"与"视频帧"的时间偏移配置
        merged_delta = {**self.get_delta_timestamps(), **self.get_video_delta_timestamps()}

        # 构造底层 LeRobotDataset,把 delta_timestamps 传进去 → __getitem__ 会据此取未来序列
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=Resize(image_size),
            delta_timestamps=merged_delta,
            load_image=load_image
        )

        self.return_item = return_item
        self.transform = transform

    def __len__(self):
        return len(self.dataset)   # 样本数 = 底层数据集帧数

    def get_features(self):
        """列出需要从 parquet 实际加载的特征列(只加载用得到的列,优化 IO)。"""
        features = set()
        for feature_category, _features in self.feature_transform.org_features.items():
            if len(self.feature_transform.actions_convert_from_state)>0 and feature_category == 'actions':
                continue   # action 由 state 推导时,不需要单独加载 action 列
            features.update(_features)
        features.update(self.feature_transform.feature_to_keep)
        features = [x for x in list(features) if x not in ['action_is_pad', 'task', 'subtask']]
        return features

    def get_delta_timestamps(self, return_indices = False):
        """★ 核心:生成 action chunk 的"时间偏移表"。
        告诉底层 LeRobotDataset:取某个 key 时,除了当前帧,还要取未来哪些帧。
        例如 chunk_size=50 → action 的偏移 = [0, 1, 2, ..., 49]/fps,即未来 50 步动作序列。
        这就是 VLA 模型预测"一段动作"而非"单步动作"的数据来源(action chunking)。
        """
        delta_timestamps = {}
        fps = None if return_indices else self.dataset_meta.fps   # return_indices=True 时用整数步索引
        if not len(self.feature_transform.actions_convert_from_state)>0:
            # 普通模式:每个动作特征取未来 chunk_size 步
            for action_feature in self.feature_transform.org_features['actions']:
                delta_timestamps[action_feature] = [t / fps if fps else t for t in range(self.chunk_size)]
        else:
            # action 由 state 推导(相对动作)模式:用 state 的 [0..chunk_size] 来推 action
            for state_feature in self.feature_transform.org_features['states']:
                delta_timestamps[state_feature] = [t / fps if fps else t for t in range(self.chunk_size+1)]
        return delta_timestamps

    def get_video_delta_timestamps(self):
        """Multi-frame time offsets for video keys; returns an empty dict when disabled."""
        """视频帧的时间偏移。use_future_image 时取 [当前, 未来] 两帧(用于 depth/video 蒸馏的未来帧);否则空。"""

        fps = self.dataset_meta.fps
        if self.use_future_image:
            offsets = [0, (self.chunk_size - 1) / fps]   # 当前帧 + chunk 末尾帧(未来)
            return {cam: offsets for cam in self.feature_transform.org_features['images']}
        else:
            return {}

    def check_lerobot_item(self, item):
        """维度规整:把 0 维标量或 1 维张量补成统一形状,避免后续 reshape 出错。"""
        # if state or action is a 0-d tensor, convert it to 1-d tensor
        if not len(self.feature_transform.actions_convert_from_state)>0:
            for action_feature in self.feature_transform.org_features['actions']:
                if len(item[action_feature].shape) == 1:
                    item[action_feature] = item[action_feature].unsqueeze(-1)   # (chunk,) -> (chunk,1)

            for state_feature in self.feature_transform.org_features['states']:
                if len(item[state_feature].shape) == 0:
                    item[state_feature] = item[state_feature].unsqueeze(-1)     # 标量 -> (1,)
        else:
            for state_feature in self.feature_transform.org_features['states']:
                if len(item[state_feature].shape) == 1:
                    item[state_feature] = item[state_feature].unsqueeze(-1)
        return item

    def getitem(self, idx):
        """★ 核心:取一条训练样本。
           底层取数 → 维度规整 → FeatureTransform.apply(字段映射 + 归一化 + padding) → 可选 transform。
        """
        raw_item = self.check_lerobot_item(self.dataset[idx])   # 底层 LeRobotDataset 取出原始序列(含 action chunk)
        if (
            self.use_future_image
            and "future_video_effective_fps" not in raw_item
            and hasattr(self, "dataset_meta")
        ):
            # 未来视频的"有效帧率":用于 video 蒸馏对齐时间
            raw_item["future_video_effective_fps"] = torch.tensor(
                float(self.dataset_meta.fps) / float(max(1, self.chunk_size - 1)),
                dtype=torch.float32,
            )
        item = self.feature_transform.apply(raw_item)   # ★ 字段映射 + 归一化 + padding 的总入口
        if self.transform is not None:
            item = self.transform(item, raw_item, self.feature_transform.feature_config.images, self.feature_transform.key_mapping)
        return item

    def __getitem__(self, idx):
        """DataLoader 实际调用的入口:直接转发到 getitem。"""
        item = self.getitem(idx)
        return item
