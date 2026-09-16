# -*- coding: utf-8 -*-
"""
multi_vla_dataset.py —— 多数据集逻辑拼接
====================================================================
作用:把多个 LeRobot 数据集(RoboTwin 50 个任务、或多个真机任务)"逻辑上"拼成一个
     大 Dataset,训练时当成一个整体来采样。无需物理合并数据文件。

核心思路:
  · 解析数据集列表文件(每行 "<robot_config_name> <数据集路径>")
  · 为每个数据集构造一个 VLADataset(底层单数据集加载器)
  · 同一个 robot_config_name 的多个数据集,共享同一个 FeatureTransform
    (映射+归一化逻辑相同,省内存且保证一致)
  · 维护各子数据集的"起始全局索引",用二分查找把全局 idx 路由到对应子数据集
  · __getitem__ 带容错重试:某帧读取失败就随机换一帧,最多 200 次,保证训练不中断
====================================================================
"""

from typing import Callable
import numpy as np
from lingbotvla.utils import helper
import torch
try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.utils.constants import HF_LEROBOT_HOME
from torchvision.transforms.v2 import Resize
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset
from .base_dataset import VLADataset

logger = helper.create_logger(__name__)

def get_all_tasks(task_files, sep=' '):
    """解析多数据集列表文件,返回 (data_names, task_list)。
    每行格式 '<robot_config_name> <数据集路径>',逗号可分隔多个 txt。"""
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
                data_names.append(data_name)   # robot config 名(如 robotwin)
                task_list.append(task)          # 数据集路径/repo_id
        f.close()
    return data_names, task_list


class MultiVLADataset(Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """
    """多个底层 VLADataset 的"逻辑拼接"。
    对外像一个统一编号的大 Dataset,内部按索引路由到各子 VLADataset。"""


    def __init__(
        self,
        repo_file: str,                 # 数据集列表文件路径(每行一个数据集)
        dataset_config,                 # 训练数据参数
        robot_config_root,              # robot config 目录
        config=None,                    # 模型 config
        processor=None,                 # VLM processor
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        video_backend: str = 'torchcodec',
        chunk_size: int = 50,
        image_size = (224, 224),
        do_nomalize = True,
        disabled_image_features = False,
        return_item = False,
        transform=None,
        prompt_type = 'both',           # 语言指令粒度:both/global/subtask
        image_augment = False,
        use_depth_align=False,
        use_future_image=False,
    ):

        self.config = config
        self.processor = processor
        self.return_item = return_item

        # 解析数据集清单:data_names(robot config 名) + repo_ids(数据集路径)
        data_names, repo_ids  = get_all_tasks(repo_file)
        self.data_names, self.repo_ids = data_names, repo_ids
        # super().__init__()
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.

        # feature_transforms:按 robot config 名缓存 FeatureTransform,
        # 同名(如多个 robotwin 任务)共享一份映射+归一化器,省内存且保证一致
        self.feature_transforms = {}

        # prompt_type 决定"用哪种语言指令粒度":
        #   both    → 每个数据集建 2 个 VLADataset(分别用 subtask / global 指令)≈ 数据增强
        #   global  → 只用全局任务指令
        #   subtask → 只用子任务指令
        if prompt_type =='both':
            use_subtask_as_prompt = [True, False]
        elif prompt_type =='global':
            use_subtask_as_prompt = [False]
        elif prompt_type =='subtask':
            use_subtask_as_prompt = [True]
        else:
            raise ValueError(f'prompt_type {prompt_type} is not supported')

        _datasets = []
        # 逐个数据集构造 VLADataset;prompt_type=both 时每个数据集会构造多个
        for i, repo_id in tqdm(enumerate(repo_ids), desc="Initializing datasets", total=len(repo_ids)):
            for _use_subtask_as_prompt in use_subtask_as_prompt:
                # 若该 robot config 已有缓存的 feature_transform 则复用,否则传 None 让 VLADataset 新建
                feature_transform = self.feature_transforms[self.data_names[i]] if self.data_names[i] in self.feature_transforms else None
                dataset = VLADataset(
                        repo_id,
                        self.data_names[i],
                        dataset_config,
                        robot_config_root,
                        config,
                        processor,
                        video_backend = video_backend,
                        chunk_size = chunk_size,
                        image_size = image_size,
                        do_nomalize = do_nomalize,
                        return_item = return_item,
                        disabled_image_features = disabled_image_features,
                        feature_transform = feature_transform,
                        transform=transform,
                        use_subtask_as_prompt = _use_subtask_as_prompt,
                        image_augment = image_augment,
                        use_depth_align = use_depth_align,
                        use_future_image=use_future_image,
                    )
                # 新建的 feature_transform 缓存起来,供后续同名数据集复用
                if self.data_names[i] not in self.feature_transforms:
                    self.feature_transforms[self.data_names[i]] = dataset.feature_transform
                _datasets.append(dataset)

        self._datasets = _datasets   # 所有子 VLADataset 的列表

        # 计算每个子数据集的"起始全局索引",用于 __getitem__ 的索引路由
        # 例如 3 个子集长度 [100, 200, 150] → start_index = [0, 100, 300]
        dataset_start_index = []
        start_index = 0
        for dataset in self._datasets:
            dataset_start_index.append(start_index)
            start_index += dataset.dataset.num_frames
        self.dataset_start_index = dataset_start_index


    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        """总帧数 = 各子数据集帧数之和。"""
        return sum(d.dataset.num_frames for d in self._datasets)


    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        """总 episode 数 = 各子数据集 episode 数之和。"""
        return sum(d.dataset.num_episodes for d in self._datasets)

    def __len__(self):
        return self.num_frames

    def getdata(self, idx: int) -> dict[str, torch.Tensor]:
        """★ 核心索引路由:把全局 idx 映射到对应子数据集内的相对 idx,再取出样本。"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        # 二分查找:idx 落在第几个子数据集(searchsorted 'right' - 1)
        dataset_idx = np.searchsorted(self.dataset_start_index,  np.array([idx]), side='right') - 1
        dataset_idx = dataset_idx[0]
        dataset = self._datasets[dataset_idx]
        # 转成子数据集内的相对索引,取出样本
        item = dataset.getitem(idx - self.dataset_start_index[dataset_idx])

        # 附上机器人标识(rep_id),方便多机器人混合训练时区分来源
        if isinstance(item, list):
            if len(item) != 1:
                raise ValueError("Expected a single item from the dataset.")
            item[0]['rep_id'] = dataset.data_name
        else:
            item['rep_id'] = dataset.data_name

        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """DataLoader 入口:带容错重试的取样本。
        多数据集合并时,个别帧可能损坏/越界,失败就随机换一帧重试,最多 200 次,
        保证训练循环不会因为单帧异常而中断。"""

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        #return self.getdata(cur)
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                # 取数失败:记录错误,随机换一个全局 idx 再试
                last_err = e
                attempts += 1
                dataset_idx = np.searchsorted(self.dataset_start_index,  np.array([cur]), side='right') - 1
                dataset_idx = dataset_idx[0]
                dataset = self._datasets[dataset_idx]
                logger.info(f"Last error: {repr(last_err)},\n"
                      f"Dataset: {dataset.dataset.repo_id}")
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )
