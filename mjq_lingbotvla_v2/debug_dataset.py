#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_dataset.py —— 调试:单独取 dataset[7],打印完整 AssertionError
(绕过 multi_vla_dataset 的重试机制,定位真正的报错位置)
"""
import sys, os, traceback

os.chdir('/home/mjq/robot_item/lingbot-vla-v2-main')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

sys.argv = [
    'debug', 'configs/vla/norm_compute/so101_norm.yaml',
    '--data.data_name=so101',
    '--data.train_path=/home/mjq/robot_item/lingbot-vla-v2-main/dual_arm_dataset',
    '--data.norm_path=assets/norm_stats/so101.json',
    '--train.output_dir=tmp_norm', '--train.max_steps=1',
]

from lingbotvla.utils.arguments import parse_args
from compute_norm_stats import Arguments, _init_dataset_worker

args = parse_args(Arguments)
args.data.chunk_size = args.train.chunk_size
args.data.data_name = 'multi'   # ★ 单数据集转 multi(和 compute_norm_stats main 一致)

# 单数据集 → 临时 txt(multi 模式)
tmp = '/tmp/debug_so101.txt'
with open(tmp, 'w') as f:
    f.write('so101 /home/mjq/robot_item/lingbot-vla-v2-main/dual_arm_dataset\n')
args.data.train_path = tmp

print('构建 dataset...')
dataset = _init_dataset_worker(args, ['so101'])
print('dataset len:', len(dataset))
print('--- 取内部 VLADataset[7](绕过 MultiVLA 重试,完整 traceback)---')
inner = dataset._datasets[0]   # 内部 VLADataset(无重试机制)
print('内部 VLADataset:', type(inner).__name__)
try:
    item = inner[7]            # 直接取 → 完整 AssertionError
    print('✅ 成功! keys:', list(item.keys()) if isinstance(item, dict) else type(item))
except Exception:
    traceback.print_exc()
