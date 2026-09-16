#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_v4_head_std.py —— v4 norm 统计的头部 std 地板(0.02)
====================================================================
【背景:为什么必须做】
  v4 把头部完全锁死(yaw=0, pitch=0.53),实测数据:
    action.head_j1 std = 0.0000(精确恒 0) → meanstd 归一化 (x-mean)/std = 0/0 = NaN
    action.Head_j2 std ≈ 1e-3(浮点抖动)   → 被除成 ±1 量级的随机噪声
  compute_norm_stats 的 RunningStats 没有 std 下限保护(只防负方差),
  所以统计算完后必须人工给头部 2 维的 std 打 0.02 下限。

【打了地板后的效果】
  头部归一化值恒 ≈0(常数/0.02);推理反归一化时即使模型预测偏 ±0.5,
  物理误差也只有 ±0.01 rad —— 头部由数据协议锁定,不依赖任何代码强制,
  rollout 侧零改动。这和当年腰部"std≈2e-4 不映射"是同一类坑的另一种解法
  (头部槽位要保留给 rollout 反查,不能像腰部那样取消映射)。

【用法】
  python mjq_lingbotvla_v2/fix_v4_head_std.py            # 干跑:只打印将改什么
  python mjq_lingbotvla_v2/fix_v4_head_std.py --apply    # 真正写回
  (文件可由 compute_norm_stats 随时重算再生,不做额外备份)
====================================================================
"""
import argparse
import json
import sys
from pathlib import Path

# ============ 配置区 ============
NORM_JSON = Path("/home/mjq/robot_item/lingbot-vla-v2-main/assets/norm_stats/h1_v4.json")
HEAD_KEYS = ("action.head.position", "observation.state.head.position")
STD_FLOOR = 0.02      # 头部 std 下限(足够小:不压缩有效动态;足够大:不被浮点噪声放大)
MIN_OTHER_STD = 1e-3  # 其余任何维度 std 低于它 = 有意外恒定维,必须人工看
# ================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写回(默认干跑)")
    args = ap.parse_args()

    d = json.loads(NORM_JSON.read_text())
    stats = d["norm_stats"]
    print(f"读入 {NORM_JSON}(count={d.get('count')})")

    # 1) 头部 4 处 std(state/action × yaw/pitch)打地板
    for k in HEAD_KEYS:
        std = stats[k]["std"]
        new = [max(v, STD_FLOOR) for v in std]
        print(f"  {k}.std: {[f'{v:.6g}' for v in std]} -> {[f'{v:.6g}' for v in new]}")
        stats[k]["std"] = new

    # 2) 断言:除头部外不允许再出现可疑恒定维
    suspicious = []
    for k, v in stats.items():
        if k in HEAD_KEYS:
            continue
        for i, s in enumerate(v["std"]):
            if s < MIN_OTHER_STD:
                suspicious.append(f"{k}[{i}] std={s:.3g}")
    if suspicious:
        sys.exit(f"[中止] 发现头部之外的恒定维,人工确认后再跑: {suspicious}")

    if not args.apply:
        print("\n[干跑] 未写回。确认后加 --apply。")
        return
    NORM_JSON.write_text(json.dumps(d, indent=2))
    print(f"\n已写回 {NORM_JSON}")

if __name__ == "__main__":
    main()
