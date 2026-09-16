# -*- coding: utf-8 -*-
"""
utils.py —— FeatureTransform:字段映射 + 归一化 + padding 引擎
====================================================================
本文件是 VLA 数据处理的"大脑",核心是 FeatureTransform 类。它把 LeRobot 取出的
原始样本(roboc 配置里定义的 origin_keys 切片)转换成模型直接消费的统一输入。

一条样本经过 apply() 的完整管线:
  原始 item
    │ ① convert_features()      原始字段切片 → 统一特征(按 robot config 拼接)
    │ ② subtract_state           相对动作:action -= state(若 subtract_state=True)
    │ ③ normalizer.normalize()   按 norm_type 归一化(meanstd / bounds_99_woclip ...)
    │ ④ pad_and_concat()         各 joint padding 到 max_dim 后拼成统一 state/action 向量
    │ ⑤ prepare_state/action/images/language  转成模型张量格式
    ▼
  batch_dict(images / state / actions / lang_tokens / masks ...) → 模型

推理时用 unapply() 反向还原:模型输出 → 反 padding → 反归一化 → 加回 state → 还原成原始动作。

关键数据结构:
  key_mapping        :正向映射 {target_feature: {origin_keys: {原始key: {start,end}}}}
  key_reverse_mapping:反向映射 {原始key: [{target_key, target_start, target_end}]}
  joint_mask         :标记哪些维度是真实数据(1)还是 padding(0),模型算 loss / 切片用
====================================================================
"""

import os
import json
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal

import ast
import torch.nn.functional as F

from ...utils import logging as logging_utils
from .transform import Normalizer, prepare_images, prepare_state, prepare_language, prepare_action, expert_visual_transform
from .ee_pose_transform import *   # 提供 relative_pose_quaternion / absolute_pose_quaternion / _is_quaternion_relative_type 等(用于 end.position 相对位姿)
from typing import Dict, List, Optional
import ast
import torch.nn.functional as F

logger = logging_utils.get_logger(__name__)

def compute_image_token_count(images, image_grid_thw=None, merge_size=2, use_vision_boundaries=True):
    """计算一批图像在 VLM 里会占用多少个 token(用于序列长度估算 / 显存)。
    若提供 image_grid_thw(Qwen3-VL 的 patch 网格),按 patch 数算;否则按图像数×patch 数估算。"""
    if not isinstance(images, torch.Tensor) or images.numel() == 0:
        return 0

    boundary_tokens = 2 if use_vision_boundaries else 0   # 每张图前后各加的边界 token
    if image_grid_thw is not None:
        grid = image_grid_thw
        if not isinstance(grid, torch.Tensor):
            grid = torch.as_tensor(grid)
        grid = grid.reshape(-1, 3)
        num_patch = grid.prod(dim=-1) // (int(merge_size) ** 2)   # (t*h*w) / merge_size²
        return int((num_patch + boundary_tokens).sum().item())

    if images.ndim == 3:
        num_images, num_patch = images.shape[0], images.shape[1]
    elif images.ndim >= 4:
        num_images, num_patch = images.shape[0], images.shape[-2]
    else:
        return 0
    return int(num_images * (num_patch + boundary_tokens))


class FeatureInfo(BaseModel):
    """从训练配置(data_config)解析出的"特征空间声明":有哪些关节类型、各自最大维度、相机名。"""
    joints: List[str] | None = None              # 关节类型名,如 ['arm.position','end.position','effector.position']
    images: List[str] | None = None              # 统一相机键名,如 ['observation.images.camera_top', ...]
    joints_max_dim: dict | None = None           # 每个关节类型的槽位维度,如 {'arm.position':14, 'effector.position':2}

    def update_info(self, data_config):
        """从 data_config.joints / cameras 填充本结构。"""
        joints_info = data_config.joints
        self.images = ['observation.images.'+image for image in data_config.cameras]

        joints= []
        joints_max_dim = {}
        for s in joints_info:
            joint_info = ast.literal_eval(s)     # yaml 里的 "{arm.position: 14}" 字符串 → dict
            joint = next(iter(joint_info.keys()))
            if joint_info[joint] == 0: continue   # 维度为 0 的跳过

            joints.append(joint)
            joints_max_dim.update(joint_info)
        self.joints = joints
        self.joints_max_dim = joints_max_dim


class FeatureTransform:
    """字段映射 + 归一化 + padding 的核心类。由 VLADataset 构造,apply() 被 VLADataset.getitem 调用。"""

    def __init__(
        self,
        robot_config_path,                # robot config yaml 路径(如 configs/robot_configs/robotwin.yaml)
        data_config,                      # 训练数据参数(joints/cameras/norm_type 等)
        model_config,                     # 模型 config(max_state_dim/max_action_dim 等)
        processor,                        # VLM processor(tokenizer + image_processor)
        disabled_image_features=False,    # 是否禁用图像特征(算 norm 时 True)
        do_nomalize=True,                 # 是否做归一化(算 norm 时 False)
        chunk_size=50,                    # 动作序列长度
        return_item_befor_padding=False,  # 是否在 padding 前返回(算 norm 时 True,只取归一化前的数值)
        norm_stats_path=None,             # norm_stats.json 路径(None 则从 robot config 读)
        use_depth_align=False,            # depth 对齐(native-depth 训练)
        image_augment=False,              # 图像增强
        use_future_image=False,):         # 加载未来帧(depth/video 蒸馏)

        # ---- 1) 读取 robot config ----
        assert os.path.exists(robot_config_path), f"{robot_config_path} does not exist."
        with open(robot_config_path, 'r') as f:
            robot_config = yaml.safe_load(f)
        f.close()

        if norm_stats_path is None:
            norm_stats_path = robot_config.pop('norm_stats')   # 从 robot config 取 norm_stats 路径并移出
        else:
            robot_config.pop('norm_stats')                      # 用外部传入的路径,但仍要把该字段从 config 移出


        # ---- 2) 解析训练配置声明的特征空间 ----
        self.feature_config = FeatureInfo()
        if getattr(data_config, 'joints', None) is not None:
            self.feature_config.update_info(data_config)

        if not return_item_befor_padding: self.check_robot_config(robot_config)   # 校验 robot config 字段合法

        self.model_config = model_config
        self.tokenizer = processor.tokenizer if processor is not None else None
        self.processor = processor

        self.chunk_size = chunk_size
        self.return_item_befor_padding = return_item_befor_padding

        # disabled_image_features: keep the image features or not when getting lerobot item
        self.disabled_image_features = disabled_image_features
        self.use_depth_align = use_depth_align
        self.use_future_image = use_future_image

        if not disabled_image_features:
            self.image_augment = image_augment

        # keep the self.feature_to_keep in lerobot item when convert to new item
        # 这些字段在 convert 时原样保留(不参与映射),多为元信息
        self.feature_to_keep = set([
            'timestamp',
            'frame_index',
            'episode_index',
            'task_index',
            'action_is_pad',
            'task',
        ])

        # ---- 3) 解析 robot config 的字段映射,得到 正向/反向 映射表 ----
        target_features  = {'states':[], 'actions':[], 'images':[]}   # 统一特征名(目标)
        org_features  = {'states':set(), 'actions':set(), 'images':set()}  # 原始字段名(来源)
        self.get_feature_mapping(robot_config, target_features, org_features)
        self.states = target_features['states']
        self.actions = target_features['actions']
        self.images = target_features['images']

        self.org_features = org_features

        # ---- 4) 构造归一化器(加载 norm_stats.json + norm_type)----
        self.normalizer = self.get_normalizer(norm_stats_path, do_nomalize, data_config)

    def get_normalizer(self, norm_stats_path, do_nomalize, data_config):
        """构造 Normalizer:把 data_config.norm_type(yaml 里每个 joint 的归一化方式)与 norm_stats.json 绑定。"""

        if not do_nomalize: return None   # 算 norm 统计时不归一化

        # norm_type 形如 ["{arm.position: bounds_99_woclip}", ...] → 展平成 {arm.position: bounds_99_woclip}
        action_state_norm_type = {k: v for d in data_config.norm_type for k, v in ast.literal_eval(d).items()}

        assert norm_stats_path is not None
        norm_type = {}
        # 给每个 action 特征分配 norm_type(按 base name 匹配,如 action.arm.position → arm.position)
        for feature in self.actions:
            base_name = feature.split('action.')[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]

        # state 特征同理
        for feature in self.states:
            base_name = feature.split('observation.state.')[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]

        # 图像不做数值归一化(后面 prepare_images 单独处理)
        for feature in self.images:
            norm_type[feature] = 'identity'

        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        f.close()

        normalizer = Normalizer(
            norm_stats=norm_stats['norm_stats'],
            norm_type=norm_type,
        )

        return normalizer


    def check_robot_config(self, robot_config):
        """校验:robot config 里出现的每个 action/state/image 特征,都必须已在训练配置的 joints/cameras 声明。"""

        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)
            for feature_convert_info in features_convert_info:

                if isinstance(feature_convert_info, dict):
                    assert len(feature_convert_info.keys()) == 1

                if feature_category == 'actions':
                    assert isinstance(feature_convert_info, dict)
                    action_feature = list(feature_convert_info.keys())[0].split('action.')[-1]
                    if action_feature not in self.feature_config.joints:
                        raise ValueError(f"{action_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}")

                elif feature_category == 'states':
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        state_feature = list(feature_convert_info.keys())[0]
                    else:
                        state_feature = feature_convert_info
                    state_feature = state_feature.split('observation.state.')[-1]
                    if state_feature not in self.feature_config.joints:
                        raise ValueError(f"{state_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}")

                elif feature_category == 'images':
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        image_feature = list(feature_convert_info.keys())[0]
                    else:
                        image_feature = feature_convert_info
                    if image_feature not in self.feature_config.images:
                        raise ValueError(f"{image_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.images}")


    def get_feature_mapping(self, robot_config, target_features, org_features):
        """★ 解析 robot config,构建"原始字段 ↔ 统一特征"的正向/反向映射表。
        产物:
          self.key_mapping        :{target_feature: {origin_keys: OrderedDict{原始key: {start,end}}}}
          self.key_reverse_mapping:{原始key: [{target_key, target_start, target_end}, ...]}
          self.action_subtract_state:{target_action: bool}   是否相对动作
          self.actions_convert_from_state:{action: state}    action 由 state 推导的特殊模式
        """
        to_convert_features = {}
        reverse_convert_features = {}
        action_subtract_state = {}
        action_relative_type = {}
        actions_convert_from_state = set()
        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)

            for feature_convert_info in features_convert_info:

                if isinstance(feature_convert_info, str):
                    # 简写形式(原始键==目标键):原样保留
                    self.feature_to_keep.add(feature_convert_info)
                    target_features[feature_category].append(feature_convert_info)
                    org_features[feature_category].add(feature_convert_info)

                elif isinstance(feature_convert_info, dict):
                    target_feature = next(iter(feature_convert_info.keys()))   # 统一特征名(目标)
                    target_features[feature_category].append(target_feature)

                    if feature_category == 'actions':
                        assert isinstance(feature_convert_info, dict)
                        assert 'subtract_state' in feature_convert_info[target_feature]
                        action_subtract_state[target_feature] = feature_convert_info[target_feature].pop('subtract_state')
                        # velocity / effector.position 不允许做 subtract_state(无意义或会出错)
                        if 'velocity' in target_feature or 'effector.position' in target_feature:
                            assert not action_subtract_state[target_feature], f"{target_feature} cannot be subtracted from state"

                        action_relative_type[target_feature] = feature_convert_info[target_feature].pop('relative_type', 'quaternion_local')
                        if_convert_from_state = feature_convert_info[target_feature].pop('convert_from_state', False)
                        if if_convert_from_state:
                            actions_convert_from_state.add(target_feature)

                        if 'origin_keys' not in feature_convert_info[target_feature] and not if_convert_from_state:
                            self.feature_to_keep.add(feature_convert_info)

                    # 处理 origin_keys(可能为 list 或 str)
                    if 'origin_keys' in feature_convert_info[target_feature]:
                        if isinstance(feature_convert_info[target_feature]['origin_keys'], list):
                            # list 形式:[{原始key: {start,end}}, ...] → 有序 dict(同名 key 加 '*' 去重)
                            ordered_origin_keys = OrderedDict()
                            for item in feature_convert_info[target_feature]['origin_keys']:
                                for k, v in item.items():
                                    while k in ordered_origin_keys:
                                        k = k+'*'
                                    ordered_origin_keys[k] = v

                            feature_convert_info[target_feature]['origin_keys'] = ordered_origin_keys
                            to_convert_features.update(feature_convert_info)

                            if feature_category in ['actions', 'states']:
                                org_features[feature_category].update([k.split('*')[0] for k in ordered_origin_keys.keys()])

                            # 计算每个原始切片在目标特征里的起止位置(用于反向还原时切片)
                            target_start_id = 0
                            for org_key, info in ordered_origin_keys.items():
                                org_info = info.copy()
                                org_key = org_key.split('*')[0]
                                if org_key not in reverse_convert_features:
                                    reverse_convert_features[org_key] = []

                                if 'start' in org_info:
                                    org_info['target_key'] = target_feature
                                    org_info['target_start'] = target_start_id
                                    org_info['target_end'] = target_start_id+org_info['end']-org_info['start']
                                    target_start_id =  org_info['target_end']
                                else:
                                    org_info['target_key'] = target_feature
                                reverse_convert_features[org_key].append(org_info)


                        if isinstance(feature_convert_info[target_feature]['origin_keys'], str):
                            # str 形式:原始键整体改名(无切片)
                            to_convert_features[target_feature] = feature_convert_info[target_feature]
                            reverse_convert_features[feature_convert_info[target_feature]['origin_keys']] = {'target_key': target_feature}
                            org_features[feature_category].add(feature_convert_info[target_feature]['origin_keys'])

        # actions_convert_from_state:某些 action 不直接采集,而是从对应 state 序列推(取 state[1:] 当 action)
        to_convert_state = {}
        if len(actions_convert_from_state)>0:
            for action in list(actions_convert_from_state):

                assert action.replace("action.", "observation.state.") in target_features['states'], f"{action} can not converted from {action.replace('action.', 'observation.state.')}"
                to_convert_state[action] = action.replace("action.", "observation.state.")
        self.actions_convert_from_state = to_convert_state

        if len(self.actions_convert_from_state)>0:
            # To keep time alignment, if any action is converted from state, all actions must be converted from state
            # 为保证时间对齐:只要有 action 由 state 推,则所有 action 都必须由 state 推
            assert len(self.actions_convert_from_state)==len(target_features['actions'])

        self.action_subtract_state = action_subtract_state
        self.action_relative_type = action_relative_type
        for feature_category, feature in org_features.items():
            if feature_category == 'actions':
                if len(feature) == 0 and len(self.actions_convert_from_state)>0:
                    org_features['actions'] == []
                    continue
            if len(feature) == 0:
                org_features[feature_category] = target_features[feature_category]
            else: org_features[feature_category] = list(feature)

        self.key_mapping = to_convert_features
        self.key_reverse_mapping = reverse_convert_features


    def convert_features(self, item, w_action):
        """★ 正向映射:按 key_mapping 把原始字段切片拼接成统一特征。
        例如 arm.position = cat([observation.state[0:6], observation.state[7:13]], dim=-1)。"""
        out_item = {}
        for target_key, convert_info in self.key_mapping.items():
            if self.disabled_image_features and target_key in self.images:
                continue   # 算 norm 时跳过图像
            if not w_action and 'action' in target_key:
                continue   # 推理(eval)时不处理 action
            if isinstance(convert_info['origin_keys'], str) and convert_info['origin_keys'] in item:
                # str 形式:整体改名
                out_item[target_key] = item[convert_info['origin_keys']]
                continue

            assert isinstance(convert_info['origin_keys'], OrderedDict)
            # list 形式:逐切片取出再拼接
            concat_list = []
            convert_success = True
            for origin_key, origin_info in convert_info['origin_keys'].items():
                origin_key = origin_key.split('*')[0]
                if origin_key not in item:
                    convert_success = False
                    break
                origin_data = item.get(origin_key)[..., origin_info['start']:origin_info['end']]   # 取切片
                concat_list.append(origin_data)
            if convert_success:
                out_item[target_key] = torch.cat(concat_list, dim=-1)   # 沿特征维拼接
            del concat_list

        # 原样保留元信息字段
        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]

        # action 由 state 推导的模式:用 state 序列的 [1:] 当 action,[0] 当当前 state
        if len(self.actions_convert_from_state)>0 and w_action:
            for action_feature in self.actions:
                state_feature = self.actions_convert_from_state[action_feature]
                assert state_feature in out_item and len(out_item[state_feature].shape) == 2
                out_item[action_feature] = out_item[state_feature][1:].clone()   # 未来 chunk 步当 action
                out_item[state_feature] = out_item[state_feature][0].clone()     # 第 0 步当当前 state
            for state_feature in self.states:
                if len(out_item[state_feature].shape) == 2:
                    out_item[state_feature] = out_item[state_feature][0]

        del item
        return out_item

    def reverse_features(self, item):
        """反向映射:统一特征 → 原始字段(用于推理输出还原成机器人可执行的原始动作)。
        与 convert_features 互逆。"""
        if len(self.actions_convert_from_state)>0:
            for action_feature, state_feature in self.actions_convert_from_state.items():
                item[state_feature] = torch.cat([item[state_feature].unsqueeze(0), item[action_feature]], dim = 0)

                item.pop(action_feature)

        out_item = {}

        for target_key, convert_info in self.key_reverse_mapping.items():
            if isinstance(convert_info, dict) and convert_info['target_key'] in item:
                # str 形式:整体改名
                out_item[target_key] = item[convert_info['target_key']]
                continue

            if isinstance(convert_info, list):
                # list 形式:从目标特征里按 target_start:target_end 切回各原始字段
                convert_info = sorted(convert_info, key=lambda x: x['end'])
                concat_list = []
                convert_success = True
                for _convert_info in convert_info:
                    if _convert_info['target_key'] not in item:
                        raise ValueError(f"{_convert_info['target_key']} is not contained in robot config as target feature")
                    concat_list.append(item[_convert_info['target_key']][..., _convert_info['target_start']:_convert_info['target_end']])
                if convert_success: out_item[target_key] = torch.cat(concat_list, dim=-1)

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]
        del item
        return out_item


    def apply(self, item, policy_eval=False):
        """★★★ 总入口:把一条原始样本转成模型输入 batch_dict。被 VLADataset.getitem 调用。
        管线:convert → subtract_state → normalize → pad_and_concat → prepare_* → batch_dict。"""
        w_action = not policy_eval   # 训练时带 action,推理(eval)时不带
        if w_action:
            # action_is_pad:标记 action chunk 里哪些时间步是 padding(episode 边界补的)
            item['action_is_pad'] = item[f"{self.org_features['actions'][0]}_is_pad"] if not len(self.actions_convert_from_state)>0 else item[f"{self.org_features['states'][0]}_is_pad"][1:]
        else:
            item['action_is_pad'] = torch.zeros(self.chunk_size)
        # ① 字段映射
        item = self.convert_features(item, w_action=w_action)


        # ② 相对动作处理:subtract_state=True 时 action 减去当前 state(学增量)
        for action_feature in self.actions:
            if self.action_subtract_state[action_feature] and w_action:
                state_feature = action_feature.replace('action.', 'observation.state.')
                if not (action_feature in item and state_feature in item):
                    raise ValueError(f"{action_feature} or/and {state_feature} are not in the item")
                relative_type = self.action_relative_type.get(action_feature)
                if _is_quaternion_relative_type(relative_type):
                    # end.position 的相对位姿用四元数表示(不是简单减法)
                    assert 'end.position' in action_feature
                    item[action_feature] = relative_pose_quaternion(
                        item[action_feature],
                        item[state_feature],
                        relative_type=relative_type,
                    )
                else:
                    item[action_feature] -= item[state_feature]

        # ③ 归一化
        if self.normalizer is not None:
            item = self.normalizer.normalize(item)

        # 算 norm 统计时,到这里就返回(padding 前的归一化值)
        if self.return_item_befor_padding:
            return item

        # ④ padding + 拼接成统一向量 + 生成 joint_mask
        batch_dict = self.pad_and_concat(item, w_action)

        # ⑤ 转成模型张量格式(state / action 截到 max_*_dim,图像预处理,语言 tokenize)
        state = prepare_state(batch_dict, self.model_config.max_state_dim)
        actions = prepare_action(batch_dict, self.model_config.max_action_dim)
        return_image_grid_thw = getattr(self.model_config, "return_image_grid_thw", False)
        if not self.disabled_image_features:
            (
                images,
                img_masks,
                pil_images,
                image_grid_thw,
                image_augment_params,
            ) = prepare_images(
                self.processor.image_processor,
                batch_dict,
                image_keys=self.feature_config.images,
                train=self.image_augment and not policy_eval,
                use_depth_align=self.use_depth_align,
                return_image_grid_thw=return_image_grid_thw,
                return_augment_params=True,
            )
            # 未来帧(用于 native-depth / video 蒸馏):复用同样的增强参数保证一致
            if self.use_future_image and len(batch_dict.get("future_image", {})) > 0:
                future_obs = {**batch_dict, "image": batch_dict["future_image"]}
                future_images, _, future_pil_images, _ = prepare_images(
                    self.processor.image_processor, future_obs,
                    image_keys=self.feature_config.images,
                    train=self.image_augment and not policy_eval,
                    use_depth_align=self.use_depth_align,
                    return_image_grid_thw=False,
                    augment_params=image_augment_params,
                )
            else:
                future_images, future_pil_images = None, None
        else:
            images, img_masks, pil_images, image_grid_thw = [], [], [], None
            future_images, future_pil_images = None, None

        # 估算图像 token 数(给模型序列长度用)
        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        image_token_count = compute_image_token_count(
            images,
            image_grid_thw=image_grid_thw,
            merge_size=merge_size,
            use_vision_boundaries=getattr(self.model_config, "qwen3vl_use_vision_boundaries", True),
        )
        batch_dict["image_token_count"] = image_token_count

        lang_tokens, lang_masks = prepare_language(self.model_config, self.tokenizer, batch_dict) # bs, seq_len
        action_is_pad = batch_dict['action_is_pad']

        # ⑥ 把 joint_mask 各自 pad 到 max_state_dim / max_action_dim(统一到模型 head 维度)
        state_joint_mask = batch_dict['state_joint_mask']
        assert self.model_config.max_state_dim >= state_joint_mask.shape[-1], f"max_action_dim is smaller than the state joint dimension: {self.model_config.max_action_dim} < {state_joint_mask.shape[-1]}"
        state_joint_mask = F.pad(state_joint_mask, (0, self.model_config.max_state_dim - state_joint_mask.shape[-1])).to(dtype=torch.bool)

        action_joint_mask = batch_dict['action_joint_mask']
        assert self.model_config.max_action_dim >= action_joint_mask.shape[-1], f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {action_joint_mask.shape[-1]}"
        action_joint_mask = F.pad(action_joint_mask, (0, self.model_config.max_action_dim - action_joint_mask.shape[-1])).to(dtype=torch.bool)

        chunk_joint_mask = batch_dict['chunk_joint_mask']
        assert self.model_config.max_action_dim >= chunk_joint_mask.shape[-1], f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {chunk_joint_mask.shape[-1]}"
        chunk_joint_mask = F.pad(chunk_joint_mask, (0, self.model_config.max_action_dim - chunk_joint_mask.shape[-1])).to(dtype=torch.bool)

        # ⑦ 组装最终 batch_dict(模型直接消费的格式)
        del batch_dict
        batch_dict = {
                'images': images,
                'img_masks': img_masks,
                'state': state,
                'lang_tokens': lang_tokens,
                'lang_masks': lang_masks,
                'actions': actions,
                'action_is_pad': action_is_pad,
                'joint_mask': chunk_joint_mask,
                'state_joint_mask': state_joint_mask,
                'action_joint_mask': action_joint_mask,
            }
        if image_grid_thw is not None:
            batch_dict['image_grid_thw'] = image_grid_thw

        if self.use_depth_align:
            batch_dict['pil_images'] = pil_images
            if self.use_future_image:
                #assert future_pil_images is not None and future_pil_images is not []:
                batch_dict['future_pil_images'] = future_pil_images
        return batch_dict

    def unapply(self, item):
        """apply 的逆:推理时把模型输出还原成原始动作空间(反 padding → 反归一化 → 加回 state → 反映射)。"""
        if not self.return_item_befor_padding:
            item = self.reverse_pad_and_concat(item)

        if self.normalizer is not None:
            item = self.normalizer.unnormalize(item)

        # 相对动作的反向:把增量加回 state(或四元数逆变换)
        for action_feature in self.actions:
            if self.action_subtract_state[action_feature]:
                state_feature = action_feature.replace('action.', 'observation.state.')
                relative_type = self.action_relative_type.get(action_feature)
                if _is_quaternion_relative_type(relative_type):
                    assert 'end.position' in action_feature
                    item[action_feature] = absolute_pose_quaternion(
                        item[action_feature],
                        item[state_feature],
                        relative_type=relative_type,
                    )
                else:
                    item[action_feature] += item[state_feature]

        item = self.reverse_features(item)
        return item

    def reverse_pad_and_concat(self, item):
        """pad_and_concat 的逆:用 joint_mask 把统一向量切回各 joint 特征(只取真实维度)。"""
        reverse_item = {}

        # In policy_eval, model output `actions` is always padded to max_action_dim
        # Pad mask with False at the tail so `actions[:, mask]` selects only the
        # real joint dims — the padded region is False and contributes nothing.
        state_joint_mask = item['state_joint_mask']
        assert state_joint_mask.shape[-1] == item['state'].shape[-1]

        action_joint_mask = item['action_joint_mask']
        assert action_joint_mask.shape[-1] == item['actions'].shape[-1]

        state = item['state'][state_joint_mask]            # 只取真实 state 维度
        action = item['actions'][:, action_joint_mask]      # 只取真实 action 维度

        # 按各 joint 的维度依次切分
        for k in self.feature_config.joints:

            state_key = f'observation.state.{k}'
            if state_key in self.states:
                joint_dim = self.normalizer.norm_stats[state_key]['mean'].shape[-1]
                reverse_item[state_key] = state[:joint_dim]
                state = state[joint_dim:]
            del state_key

            action_key = f'action.{k}'
            if action_key in self.actions:
                joint_dim = self.normalizer.norm_stats[action_key]['mean'].shape[-1]
                reverse_item[action_key] = action[:, :joint_dim]
                action = action[:, joint_dim:]
            del action_key
        return reverse_item

    def pad_and_concat(self, item, w_action = True):
        """★ 把各 joint 特征按 joints_max_dim padding 后,拼接成统一的 state/action 向量,并生成 joint_mask。
        这一步就是把"各机器人占用的特征子集"对齐到统一的 max 维度空间(对应 55 维统一动作空间)。"""
        images = {}
        future_images = {}
        for image_key in self.feature_config.images:
            if image_key in self.images and image_key in item:
                if not self.use_future_image:
                    images[image_key] = item[image_key]
                else:
                    # use_future_image:第 0 帧当当前,最后一帧当未来
                    images[image_key] = item[image_key][0]
                    future_images[image_key] = item[image_key][-1]

        actions, action_joints_pad = [], []
        states, state_joints_pad = [], []

        # 按训练配置声明的 joint 顺序(arm/end/effector/...),逐个 padding 后拼接
        for k in self.feature_config.joints:
            state_key = f'observation.state.{k}'

            if state_key in self.states:
                # padding 到该 joint 的槽位维度 joints_max_dim[k](不足补零)
                pad_len = self.feature_config.joints_max_dim[k] - item[state_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                states.append(F.pad(item[state_key], (0, pad_len)))
                state_joints_pad.append(F.pad(torch.ones(item[state_key].shape), (0, pad_len)))   # 真实维度=1
            else:
                # 该机器人没有这个 joint:全零 + mask 全 0
                states.append(torch.zeros(self.feature_config.joints_max_dim[k]))
                state_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del state_key

            action_key = f'action.{k}'
            if action_key in self.actions and w_action:
                # 训练时:action chunk padding
                pad_len = self.feature_config.joints_max_dim[k] - item[action_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                actions.append(F.pad(item[action_key], (0, pad_len)))
                action_joints_pad.append(F.pad(torch.ones(item[action_key].shape[-1]), (0, pad_len)))
            elif action_key in self.actions and not w_action:
                # 推理时:action 占位全零(模型会填),mask 按真实维度
                assert action_key in self.normalizer.norm_stats, f"{action_key} not in norm keys: {self.normalizer.norm_stats.keys()}"
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                pad_len = self.feature_config.joints_max_dim[k] - self.normalizer.norm_stats[action_key]['mean'].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                action_joints_pad.append(F.pad(torch.ones(self.normalizer.norm_stats[action_key]['mean'].shape[-1]), (0, pad_len)))
            else:
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                action_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del action_key

        # 拼接 + 生成三种 mask
        action_joint_mask = torch.cat(action_joints_pad, dim=-1).to(dtype=torch.bool)   # (max_action_dim,) 真实 action 维
        state_joint_mask = torch.cat(state_joints_pad, dim=-1).to(dtype=torch.bool)      # (max_state_dim,) 真实 state 维
        state = torch.cat(states, dim=-1).to(torch.float32)                              # 统一 state 向量
        action = torch.cat(actions, dim=-1).to(torch.float32)                            # (chunk, max_action_dim)
        chunk_joint_mask = action_joint_mask.clone().unsqueeze(0).repeat(self.chunk_size, 1)  # 扩展到 chunk 维

        batch_dict =  {
            "image": images,
            "future_image": future_images,
            "state": state,
            "action": action,
            "action_is_pad": item['action_is_pad'],
            'chunk_joint_mask': chunk_joint_mask,
            "action_joint_mask": action_joint_mask,
            "state_joint_mask": state_joint_mask,
            "prompt": [item["task"]],   # 语言指令(任务描述文本)
        }
        if "future_video_effective_fps" in item:
            batch_dict["future_video_effective_fps"] = item["future_video_effective_fps"]

        return batch_dict
