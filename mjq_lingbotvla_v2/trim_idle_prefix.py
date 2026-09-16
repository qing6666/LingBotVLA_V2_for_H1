#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trim_idle_prefix.py —— 裁掉每条轨迹开头的"静止前缀",生成新数据集副本
====================================================================
【背景:为什么要裁】
  部署时启动失败(开局抖动不走)。实测 101 条数采轨迹:开头双臂静止
  中位 16 帧 / 均值 23 帧 / 最长 103 帧(头部云台从第 0 帧就在动,但
  手臂没动),导致"开局 home 位姿 + 桌面完整场景"这个视觉区域的动作
  标签几乎 100% 是"stay"。部署恰恰从这个区域出发 → 模型学会"别动"。
  裁掉静止前缀后,episode 第 0 帧的视觉 ≈ home、标签 = 动作已起步。

【怎么做:零视频重编码】
  LeRobot v3 把多条 episode 连续存进同一个 mp4,episode 元数据表
  (meta/episodes/)记录每条在该 mp4 里的 from_timestamp;训练加载器
  (lingbotvla/data/vla_data/base_dataset.py:_query_videos)按
  "from_timestamp + 行timestamp" 抽帧。所以只需改三处元数据:
    1) data parquet   : 丢掉每条开头 onset 行,重编 index/frame_index/
                        timestamp/episode_index
    2) episodes 表    : from_timestamp += onset/fps(视频里的新起点),
                        length / dataset_from/to_index 同步更新
    3) meta/info.json : total_episodes / total_frames 更新
  videos/ 目录原样复制 —— mp4 一字节不动,零重编码、零画质损失。
  ★ 原数据集全程只读;输出到 DST 新目录,两者互不影响。

【onset 判定】
  手臂 14 维(state[2:16],腰 2 维和头 2 维不算)相邻帧差 max
  > EPS(0.004 rad) 的首帧。整条从未超过阈值的 episode 判为废数据
  整条丢弃;裁后不足 MIN_KEEP 帧的也丢弃。

【用法】
  python mjq_lingbotvla_v2/trim_idle_prefix.py             # dry-run:只统计不写盘
  python mjq_lingbotvla_v2/trim_idle_prefix.py --apply     # 真正生成副本
  python mjq_lingbotvla_v2/trim_idle_prefix.py --apply --force  # 目标已存在则覆盖
====================================================================
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ============ 解释器自纠正保险(必须在 import numpy/pyarrow 之前) ============
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}", flush=True)
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# ============================================================================

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ==================== 配置(改成你的参数)====================
# v4 批次(头部完全固定版,2026-08-27 收官 302 回合全量 QC 通过,无人工剔除):
# 全量 QC 结论 —— 结构/NaN/域/夹爪沿/风格/loader 全绿,ep213"双闭合"为 0.6 阈值
# 抖动误报(一次连续闭合),保留;17 条 L 闭合时刻离群只是节奏慢,事件数正常。
# (v3 批次时这里曾是 {30,33,36,37,100,141},依据见 v3 数据目录 DROP_LIST.txt)
SRC = Path("/home/mjq/robot_item/lingbot-vla-v2-main/H1_simulation_model/H1_build/data/h1_build_pick_v4")
DST = SRC.parent / "h1_build_pick_v4_trimmed"   # 输出目录(另起一个,原数据不动)
EPS = 0.004        # onset 阈值(rad/帧):手臂相邻帧差超过它算"动了"
ARM_SLICE = (2, 16)  # 手臂 14 维在 20 维 state 里的切片(左 7 + 右 7)
MIN_KEEP = 100     # 裁完后不足这么多帧的 episode 整条丢弃
DROP_EPISODES = set()   # v4 全量 QC 无剔除
CAM_KEYS = ["observation.images.head_rgb",
            "observation.images.left_wrist_rgb",
            "observation.images.right_wrist_rgb"]
# =============================================================


def load_all():
    """读原数据集的 info / episodes 表 / data 表(全部只读)。"""
    info = json.loads((SRC / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    ep_files = sorted((SRC / "meta" / "episodes" / "chunk-000").glob("*.parquet"))
    ep_tables = [pq.read_table(f) for f in ep_files]          # 每个文件一个 pyarrow 表
    eps = pd.concat([t.to_pandas() for t in ep_tables], ignore_index=True)

    data_files = sorted((SRC / "data" / "chunk-000").glob("*.parquet"))
    data_tables = [pq.read_table(f) for f in data_files]
    df = pd.concat([t.to_pandas() for t in data_tables], ignore_index=True)
    return info, fps, ep_files, ep_tables, eps, data_files, data_tables, df


def compute_onsets(eps, df, fps):
    """逐 episode 算静止前缀长度 onset;返回 {episode_index: (onset|None, length)}。
    onset=None 表示整条从未动过(废数据)。"""
    a0, a1 = ARM_SLICE
    res = {}
    for eid, g in df.groupby("episode_index", sort=True):
        s = np.stack(g["observation.state"].apply(lambda x: np.frombuffer(x, dtype=np.float32)))
        diff = np.abs(np.diff(s[:, a0:a1], axis=0)).max(axis=1)   # 每帧手臂最大变化
        moved = np.nonzero(diff > EPS)[0]
        # diff[i] = s[i+1]-s[i] → 首个"动"的帧号是 moved[0]+1,静止前缀 = moved[0]+1 帧
        onset = int(moved[0] + 1) if len(moved) else None
        res[int(eid)] = (onset, len(g))
    assert len(res) == len(eps), "episode 数和 episodes 表不一致"
    return res


def preflight(eps, df, onsets):
    """裁剪前断言:布局假设必须全部成立,否则宁可不做。"""
    ids = eps["episode_index"].to_numpy()
    assert (ids == np.arange(len(ids))).all(), "episode_index 不是 0..N 连续!"
    assert (df["index"].to_numpy() == np.arange(len(df))).all(), "index 列不是 0..N-1!"
    cnt = df.groupby("episode_index").size().reindex(ids).to_numpy()
    assert (cnt == eps["length"].to_numpy()).all(), "每 episode 行数 ≠ length!"
    for _, r in eps.iterrows():
        assert r["dataset_to_index"] - r["dataset_from_index"] == r["length"]
        for c in CAM_KEYS:
            f_ts = r[f"videos/{c}/from_timestamp"]
            assert abs(f_ts * 30 - round(f_ts * 30)) < 1e-3, f"from_ts 不在 1/30 网格: ep{_}"
    print("[预检] 布局假设全部通过(episodes 连续 / index 连续 / from_ts 在 1/30 网格)")


def report(onsets, keep_plan, total_frames_old, n_eps_old):
    """打印裁剪统计。"""
    onsets_kept = sorted(o for o, _ in keep_plan.values())
    dropped = sorted(eid for eid, (o, ln) in onsets.items() if o is None)
    too_short = sorted(eid for eid, (o, ln) in onsets.items()
                       if o is not None and ln - o < MIN_KEEP)
    total_cut = sum(o for o, _ in keep_plan.values())
    total_frames_new = sum(ln - o for o, ln in keep_plan.values())
    print("\n===== 裁剪统计(dry-run 与 --apply 共用)=====")
    print(f"原数据: {n_eps_old} episodes / {total_frames_old} 帧")
    q = np.percentile(onsets_kept, [10, 50, 90]) if onsets_kept else [0, 0, 0]
    print(f"保留 {len(keep_plan)} 条的 onset(帧): min={min(onsets_kept)} "
          f"p10={q[0]:.0f} 中位={q[1]:.0f} 均值={np.mean(onsets_kept):.1f} "
          f"p90={q[2]:.0f} max={max(onsets_kept)}")
    if dropped:
        lens = [onsets[e][1] for e in dropped]
        print(f"整条丢弃(全程静止): {len(dropped)} 条 {dropped},长度 {lens}")
    if too_short:
        print(f"整条丢弃(裁后不足 {MIN_KEEP} 帧): {len(too_short)} 条 {too_short}")
    if DROP_EPISODES:
        print(f"人工剔除(质检清单 DROP_LIST.txt,整条跳过): "
              f"{len(DROP_EPISODES)} 条 {sorted(DROP_EPISODES)}")
    print(f"新数据: {len(keep_plan)} episodes / {total_frames_new} 帧 "
          f"(裁掉 {total_cut} 帧 = {total_cut/total_frames_old:.1%},丢弃静止整条 "
          f"{total_frames_old - total_cut - total_frames_new} 帧)")
    return dropped, too_short


def apply_trim(info, fps, ep_files, ep_tables, data_files, data_tables, onsets, keep_plan):
    """真正生成 DST 副本。"""
    if DST.exists():
        shutil.rmtree(DST)                       # --force 已在 main 里确认过
    print(f"[复制] {SRC} -> {DST}(videos 2.2G 原样复制,mp4 不动)...")
    shutil.copytree(SRC, DST)

    # 旧 id -> 新 id(保留顺序连续重编号;lerobot 的 episodes 列表按位置索引,不能留空洞)
    old2new = {eid: i for i, eid in enumerate(keep_plan)}
    o_map = {eid: onsets[eid][0] for eid in keep_plan}

    # ---- 1) 重写 data parquet:丢前缀行 + 重编 4 列 ----
    print("[重写] data/*.parquet ...")
    gidx = 0                                       # 全局行计数器(按文件顺序推进)
    for f, tbl in zip(data_files, data_tables):
        ep_col = tbl.column("episode_index").to_numpy()
        fi_col = tbl.column("frame_index").to_numpy()
        keep_rows = [i for i, (eid, fi) in enumerate(zip(ep_col, fi_col))
                     if eid in keep_plan and fi >= o_map[eid]]
        new_tbl = tbl.take(keep_rows)
        ke = ep_col[keep_rows]
        kf = fi_col[keep_rows] - np.array([o_map[e] for e in ke])          # 新 frame_index 从 0 起
        n_rows = len(keep_rows)
        schema = tbl.schema

        def setcol(t, name, values):
            i = schema.get_field_index(name)
            return t.set_column(i, name, pa.array(values, type=schema.field(i).type))

        new_tbl = setcol(new_tbl, "episode_index", [old2new[e] for e in ke])
        new_tbl = setcol(new_tbl, "frame_index", kf)
        new_tbl = setcol(new_tbl, "timestamp", (kf / fps).astype(np.float32))  # 与原生成方式一致
        new_tbl = setcol(new_tbl, "index", np.arange(gidx, gidx + n_rows, dtype=np.int64))
        gidx += n_rows
        pq.write_table(new_tbl, DST / f.relative_to(SRC), compression="snappy")
    total_frames_new = gidx

    # ---- 2) 重写 episodes 表:length / index 范围 / from_timestamp ----
    print("[重写] meta/episodes/*.parquet ...")
    new_from = {}                                   # 新 id -> dataset_from_index
    run = 0
    for eid in keep_plan:
        new_from[old2new[eid]] = run
        run += onsets[eid][1] - o_map[eid]
    for f, tbl in zip(ep_files, ep_tables):
        ep_col = tbl.column("episode_index").to_numpy()
        keep_rows = [i for i, e in enumerate(ep_col) if e in keep_plan]
        new_tbl = tbl.take(keep_rows)
        ke = [int(e) for e in ep_col[keep_rows]]
        schema = tbl.schema
        o_arr = np.array([o_map[e] for e in ke], dtype=np.float64)

        def setcol(t, name, values):
            i = schema.get_field_index(name)
            return t.set_column(i, name, pa.array(values, type=schema.field(i).type))

        new_tbl = setcol(new_tbl, "episode_index", [old2new[e] for e in ke])
        new_tbl = setcol(new_tbl, "length",
                         [onsets[e][1] - o_map[e] for e in ke])
        new_tbl = setcol(new_tbl, "dataset_from_index",
                         [new_from[old2new[e]] for e in ke])
        new_tbl = setcol(new_tbl, "dataset_to_index",
                         [new_from[old2new[e]] + onsets[e][1] - o_map[e] for e in ke])
        for c in CAM_KEYS:                          # 每相机各自平移(它们 from 本就不同步)
            i = schema.get_field_index(f"videos/{c}/from_timestamp")
            old_ts = np.array(new_tbl.column(i).to_pylist(), dtype=np.float64)
            # 在 1/30 网格上整数帧平移,避免浮点累积误差
            new_ts = (np.round(old_ts * fps) + o_arr) / fps
            new_tbl = new_tbl.set_column(i, f"videos/{c}/from_timestamp",
                                         pa.array(new_ts, type=schema.field(i).type))
        pq.write_table(new_tbl, DST / f.relative_to(SRC), compression="snappy")

    # ---- 3) info.json 总量 ----
    info["total_episodes"] = len(keep_plan)
    info["total_frames"] = total_frames_new
    (DST / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[重写] meta/info.json: {len(keep_plan)} episodes / {total_frames_new} 帧")

    # ---- 4) 写回后自校验 ----
    print("\n[自校验] 重新读回副本核对账目 ...")
    info2 = json.loads((DST / "meta" / "info.json").read_text())
    eps2 = pd.concat([pq.read_table(f).to_pandas() for f in
                      sorted((DST / "meta" / "episodes" / "chunk-000").glob("*.parquet"))],
                     ignore_index=True)
    df2 = pd.concat([pq.read_table(f).to_pandas() for f in
                     sorted((DST / "data" / "chunk-000").glob("*.parquet"))], ignore_index=True)
    assert len(df2) == info2["total_frames"] == total_frames_new, "总帧数对不上!"
    assert len(eps2) == info2["total_episodes"] == len(keep_plan), "episode 数对不上!"
    assert (df2["index"].to_numpy() == np.arange(len(df2))).all(), "新 index 不连续!"
    assert (eps2["episode_index"].to_numpy() == np.arange(len(eps2))).all(), "新 episode_index 不连续!"
    for _, r in eps2.iterrows():
        g = df2[df2.episode_index == r["episode_index"]]
        assert len(g) == r["length"], "行长不符"
        assert r["dataset_to_index"] - r["dataset_from_index"] == r["length"]
        assert g["frame_index"].tolist() == list(range(r["length"]))
        for c in CAM_KEYS:                          # 新 from 仍在 1/30 网格上
            v = r[f"videos/{c}/from_timestamp"]
            assert abs(v * fps - round(v * fps)) < 1e-3, "新 from_ts 脱离网格!"
    print("[自校验] 全部通过 ✓")
    return total_frames_new


def main():
    ap = argparse.ArgumentParser(description="裁掉数采轨迹开头静止前缀,生成新数据集副本")
    ap.add_argument("--apply", action="store_true", help="真正写盘(默认 dry-run 只统计)")
    ap.add_argument("--force", action="store_true", help="DST 已存在时删除重建")
    args = ap.parse_args()

    info, fps, ep_files, ep_tables, eps, data_files, data_tables, df = load_all()
    onsets = compute_onsets(eps, df, fps)
    preflight(eps, df, onsets)

    keep_plan = {eid: (o, ln) for eid, (o, ln) in onsets.items()
                 if o is not None and ln - o >= MIN_KEEP
                 and eid not in DROP_EPISODES}
    dropped, too_short = report(onsets, keep_plan, len(df), len(eps))

    if not args.apply:
        print("\n[dry-run] 未写盘。确认无误后加 --apply 生成副本。")
        return
    if DST.exists() and not args.force:
        sys.exit(f"[中止] {DST} 已存在;确认覆盖请加 --force")
    apply_trim(info, fps, ep_files, ep_tables, data_files, data_tables, onsets, keep_plan)
    print(f"\n完成 ✓ 原数据(只读): {SRC}\n     新副本:          {DST}")


if __name__ == "__main__":
    main()
