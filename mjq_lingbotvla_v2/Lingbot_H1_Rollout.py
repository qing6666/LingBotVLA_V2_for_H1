#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lingbot_H1_Rollout.py —— LingBot-VLA 2.0 H1 仿真闭环推理(rollout)程序
====================================================================
【这个脚本做什么】
  用训练好的 LingBot-VLA checkpoint 闭环控制 MuJoCo 里的 H1 双臂,完成
  "Pick up the red cube and put it into the green bin",统计成功率。
  参照 h1_smolvla_inference_rollout.py(SmolVLA 版)改写:环境侧(MuJoCo 场景/
  观测渲染/动作执行/成功判定)整套复用,策略侧换成 LingBot 的推理服务
  deploy/lingbot_vla_v2_policy.LingbotVLAv2Server(与开环评估同一个类,
  也是官方 websocket 部署服务背后的类 —— 这就是"部署同款推理路径")。

【与训练/数采的三条对齐原则】(改任何一条都会让策略表现崩坏)
  1. 观测组装与 h1_dataset_recorder.record_frame 逐位一致:
     三相机图像 + state 在同一 MuJoCo 仿真时刻采样;
     state 20 维 = 上身 18 关节 qpos + 夹爪 actuator_length(0..1);
     图像 uint8 HWC 512×512。
  2. 观测键名用数据集原生名(observation.images.head_rgb / observation.state),
     FeatureTransform 会按 robot_config(h1.yaml)自动完成 20维→55维映射+归一化;
     任务文本放 'task' 键(与数据集 meta 的任务串同一句)。
  3. 动作执行与采集时定义一致:模型吐出 20 维数据集口径动作
     (18 关节位置 ctrl + 2 夹爪开合 ctrl);其中腰部 [0:2] 未参与训练映射
     (数据里全程静止),执行时钉在 home 值,只用 [2:20]。

【chunk 机制】(与 smolvla 的 select_action 队列同款语义)
  LingbotVLAv2Server(use_length=50, chunk_ret=False):
  每帧调用 infer() 拿单步动作;内部只在 global_step % 50 == 0 时才真正
  前向一次(30Hz 下约 1.67s 推理一次,bf16 单次约 0.34s)。
  --reinfer-every K 可改成"每 K 帧重推一次"(滑动窗口/回收视野,常能提成功率,
  但偏离训练时的 chunk 语义,默认 50 = 与训练完全一致)。

【回合流程】(复刻遥操作数采的 B→A→X 节奏)
  mj_resetDataKeyframe(home) → 保持 home ctrl 静置 0.6s(方块落稳)
  → policy.reset() 清空动作队列 → 30Hz 闭环 → 超帧或双块入桶即结束。
  成功判定:方块中心 |x-0.7|<0.03 且 |y|<0.03 且 0.646<z<0.75。

【怎么跑】(lingbotv2 环境;已装 mujoco==3.8.0,与数采环境同款)
  ★ VSCode:打开本文件点右上角 ▶(默认:仿真窗口+左右腕小窗+推理监视窗,30Hz 实时)
  ★ 终端:
    # 观看模式:实时看推理(仿真窗口 + 腕部小窗 + 推理监视窗)
    python mjq_lingbotvla_v2/Lingbot_H1_Rollout.py
    # 无头批量统计成功率
    python mjq_lingbotvla_v2/Lingbot_H1_Rollout.py --headless --episodes 10
    # 常用参数
    #   --checkpoint .../global_step_20000/hf_ckpt   换档对比
    #   --episodes 20 / --max-steps 1800             批量/限时(观看默认 1800 帧=60s)
    #   --reinfer-every 25                           每 25 帧重推理(实验项)
    #   --no-monitor                                 关掉推理监视窗(只要仿真画面)

【推理监视窗】("LingBot Inference Monitor")实时展示模型在想什么:
  * 每行一个关节(左臂7/右臂7/头2/夹爪2,共 18 行):
      白线=机器人当前 state 历史;黄线=模型逐步输出的 action 历史;
      右侧青色粗线=最近一次前向规划的 50 步动作块(模型的"计划书")
  * 顶栏:当前帧号 / 前向次数 / 单次前向耗时 / 实际帧率 / 任务文本
  * 任何窗口按 q 提前结束本回合,Ctrl+C 退出
====================================================================
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ============ 解释器自纠正保险(同训练/评估启动器) ============
# ★必须放在 import numpy/cv2/torch 之前:base 环境没有这些包,
#   放在后面的话,import 先炸,自纠正永远轮不到执行(只依赖 stdlib)
_LINGBOTV2_PY = "/home/mjq/miniconda3/envs/lingbotv2/bin/python"
if sys.executable != _LINGBOTV2_PY and Path(_LINGBOTV2_PY).exists():
    print(f"[解释器自纠正] {sys.executable}\n              -> {_LINGBOTV2_PY}", flush=True)
    os.execv(_LINGBOTV2_PY, [_LINGBOTV2_PY] + sys.argv)
# =============================================================

import numpy as np

# 项目根目录(本脚本在 mjq_lingbotvla_v2/ 下)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
H1_ROOT = PROJECT_ROOT / "H1_simulation_model" / "H1_build"


# ── 参数解析先于 import mujoco:无头模式强制 EGL 离屏渲染 ──────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LingBot-VLA 2.0 H1 MuJoCo closed-loop rollout")
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "output/h1_v4_full/checkpoints/global_step_40000/hf_ckpt"),
                        help="hf_ckpt 目录(默认=B方案全模型@40000 终档;"
                             " 对比用 output/h1_v4_full/checkpoints/global_step_20000/hf_ckpt;"
                             " 旧方案A用 output/h1_v4_w3/checkpoints/global_step_152000/hf_ckpt)")
    parser.add_argument("--headless", action="store_true", help="无窗口全速批量评估(默认开仿真窗口+腕部小窗+推理监视窗实时观看)")
    parser.add_argument("--episodes", type=int, default=0, help="评估回合数(默认:观看 1 回合 / 无头 10 回合)")
    parser.add_argument("--max-steps", type=int, default=0, help="每回合最大帧数(默认一律 1800=60s;演示左爪闭合在 ~36s,更短的窗口会把左块结构性判死 —— v4_w 评测踩过的坑)")
    parser.add_argument("--no-monitor", action="store_true", help="观看模式关掉推理监视窗(只留仿真窗口+腕部小窗)")
    parser.add_argument("--task", type=str, default="", help="任务指令(默认取数采记录器里的 DEFAULT_TASK)")
    parser.add_argument("--settle", type=float, default=0.6, help="回合开始前的静置秒数(等方块落稳)")
    parser.add_argument("--reinfer-every", type=int, default=50,
                        help="每多少帧重新推理一次(默认 50=整块执行完再重推,与训练 chunk 语义一致;"
                             "改小(如 10/25)=滑动窗口重推,常能提成功率但偏离训练口径)")
    parser.add_argument("--norm-path", type=str, default=str(PROJECT_ROOT / "assets/norm_stats/h1_v4.json"),
                        help="归一化统计表(必须与训练时同一份!默认=v4 批次配套的 h1_v4.json,"
                             "头部 std 已打 0.02 地板;评 v3 时才传 h1_v3.json)")
    parser.add_argument("--bias-comp", type=str, default="off",
                        choices=["off", "right", "all", "v3right"],
                        help="执行侧偏置补偿确诊实验(默认 off=不加,完全不影响现有行为)。"
                             "right=只补右臂 R_J1−0.142/R_J4+0.188;all=左右臂都补;"
                             "数值=20000 档逐关节诊断(/tmp/diag_arm_dims.py)实测偏置取负,逐帧加在执行动作上")
    parser.add_argument("--action-gain", type=float, default=1.0,
                        help="动作幅值增益确诊实验(默认 1.0=不生效):执行 a'=state+λ(action−state),"
                             "把每帧命令位移相对当前状态放大 λ 倍。针对预测幅值系统性偏小、闭环停在"
                             "目标内上方的缩短平衡点(左右臂对称偏中线+上方即此症);建议从 1.2/1.3 试起")
    parser.add_argument("--lead-clip", type=float, default=0.0,
                        help="命令超前限幅确诊实验(默认 0=不生效):逐帧把 |action−state| 逐维钳到该值"
                             "(rad)。针对 2026-09-01 实测病灶——终段相位模型输出的 chunk 首帧命令"
                             "离当前状态 0.7-0.8 rad(数据口径中位仅 0.024),每 50 帧边界 teleport "
                             "一次,画面表现为周期性大跳变+一抽一抽。限幅后臂以有界速度追模型意图;"
                             "数据里命令超前 p99≈0.28,建议从 0.15 试起")
    parser.add_argument("--grip-log", type=str, default="",
                        help="爪相位诊断:逐帧记录(帧,爪cmd L/R,指端-左右块距离)追加写 CSV;"
                             "回答'贴近方块时爪命令到底闭不闭'——不闭=缺抓取技能(训练侧解决),"
                             "闭但时机错=相位偏移(执行侧可救)")
    parser.add_argument("--spawn", type=str, default="home", choices=["home", "band", "v3"],
                        help="回合开始方块出生位。home=keyframe 固定点(=v3 老固定点 0.65,±0.30,"
                             "与数采 A 键同源);v3=与 home 同一点(历史兼容保留,写法显式);"
                             "band=随机出生带(备用,遥操作实测部分落点不好抓,已弃用)")
    parser.add_argument("--smooth", type=str, default="off", choices=["off", "ruckig"],
                        help="动作平滑层(默认 off=完全不影响现有行为)。ruckig=推理输出先过"
                             " Ruckig 在线轨迹生成(限速/限加速度/限 jerk 的 S 型过渡)再执行。"
                             "治 chunk 边界 teleport 的抽动(根因=模型 chunk 首帧超前 0.16-0.49"
                             " rad,数采 p99 仅 0.007-0.031 rad/帧);只治表现不治决策(方案文档 §19)")
    args = parser.parse_args()
    if args.episodes <= 0:
        args.episodes = 10 if args.headless else 1
    if args.max_steps <= 0:
        args.max_steps = 1800
    return args


ARGS = parse_args()
if ARGS.headless:
    os.environ.setdefault("MUJOCO_GL", "egl")  # 无头离屏渲染(本机已验证 egl 可用)

# cv2 必须先于 mujoco 导入(遥操作同款顺序,反序会卡死在窗口创建)
import cv2  # noqa: E402
import mujoco  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))                      # import deploy/lingbotvla 用
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # noqa: E402

# 关节/夹爪/相机常量直接从数采记录器 import —— 单一事实来源,杜绝顺序漂移
sys.path.insert(0, str(H1_ROOT / "teleop"))
from h1_dataset_recorder import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_TASK,
    GRIPPER_ACTUATOR_NAMES,
    UPPER_BODY_JOINTS,
)
from h1_spawn_zones import apply_cube_spawns, apply_fixed_spawns, spawns_str  # noqa: E402  # v4 出生点工具(与数采同源)

# ==================== 配置区 ====================
MJCF_PATH = H1_ROOT / "mujoco" / "H1_scene.xml"
FPS = 30
IMAGE_SIZE = 512
ROBO_NAME = "h1"                       # robot_config 名(与训练一致)

# ---- 执行侧偏置补偿(确诊实验,--bias-comp 开启;默认关)----
# 20 维动作布局:[0:2]腰 [2:9]左臂 [9:16]右臂 [16:18]头 [18:20]夹爪。
# 数值来源:h1_trim/global_step_20000 开环逐关节诊断的实测偏置(预测−真值均值),
# 补偿=偏置取负。夹爪不补:其偏置(−0.07/−0.12)本来就偏向"闭",再补会往开推,
# 与"闭合迟/浅"的病灶相反;Head_j2 偏置跨 run 不稳定,也不补。
BIAS_COMP_PRESETS = {
    "right": {9: -0.1424, 12: +0.1880},              # dim9=R_J1, dim12=R_J4
    "all":   {9: -0.1424, 12: +0.1880,               # 右臂
              2: +0.0841, 5: +0.1587},               # dim2=L_J1, dim5=L_J4
    # v3 模型实测偏置(2026-08-25 diag_arm_dims_v3,148 条 8 轨迹):
    # R_J1 +0.191 / R_J4 −0.163 / Grip.R −0.260(全 chunk 平坦,提前闭爪)
    "v3right": {9: -0.1908, 12: +0.1631, 19: +0.26},  # dim19=Grip.R
}

CUBE_BODY_NAMES = ("red_cube_left", "red_cube_right")
BIN_CENTER_XY = (0.7, 0.0)             # green_bin body 位置
BIN_XY_TOL = 0.03                      # 内腔半宽 0.045,容差收紧到 0.03(贴墙不算)
BIN_Z_RANGE = (0.646, 0.75)            # 底面 0.646,壁顶 0.741,方块静止约 0.666

# ---- Ruckig 平滑层(--smooth ruckig)的运动限值 ----
# 来源:数采 parquet 逐维差分实测(2026-09-15,302 回合)× 2-3 倍余量:
#   臂 |Δ|p99 = 0.007-0.031 rad/帧(0.22-0.92 rad/s),|ΔΔ|p95 ≈ 1.9-9.3 rad/s²;
#   爪 |Δ|p99 = 0.06 rad/帧(1.8 rad/s,数采 3-5 帧内完成开闭);
#   腰/头数采全程静止(给小值防漂移)。只拦 chunk 边界 teleport(0.16-0.49
#   rad/帧 ≈ 4.8-14.7 rad/s),不拖慢演示速度的运动。
SMOOTH_LIMITS = {"arm": (2.0, 20.0, 200.0),    # (max_vel, max_acc, max_jerk)
                 "waist": (1.0, 10.0, 100.0),
                 "head": (1.0, 10.0, 100.0),
                 "grip": (4.0, 40.0, 400.0)}


class RuckigSmoother:
    """每个 30Hz 控制拍:target=策略动作 → Ruckig 给出满足限值的下一拍位置。
    内部自持 pos/vel/acc(pass_to_input 前滚),对 chunk 边界跳变做 S 型过渡;
    target 每帧更新 = 在线重规划,不会"停下再跳"(方案文档 §13)。"""

    def __init__(self, dofs: int = 20, dt: float = 1.0 / FPS):
        from ruckig import InputParameter, OutputParameter, Ruckig
        self.otg = Ruckig(dofs=dofs, delta_time=dt)
        self.inp = InputParameter(dofs)
        self.out = OutputParameter(dofs)
        group_of = lambda i: ("grip" if i >= 18 else "head" if i >= 16
                              else "waist" if i < 2 else "arm")
        groups = [group_of(i) for i in range(dofs)]
        self.inp.max_velocity = [SMOOTH_LIMITS[g][0] for g in groups]
        self.inp.max_acceleration = [SMOOTH_LIMITS[g][1] for g in groups]
        self.inp.max_jerk = [SMOOTH_LIMITS[g][2] for g in groups]
        self.inp.target_velocity = [0.0] * dofs     # 到点即停
        self.inp.target_acceleration = [0.0] * dofs
        self.ready = False

    def reset(self, q: np.ndarray) -> None:
        """每回合开始:从当前真实状态零速启动。"""
        self.inp.current_position = list(map(float, q))
        self.inp.current_velocity = [0.0] * len(q)
        self.inp.current_acceleration = [0.0] * len(q)
        self.ready = True

    def step(self, target: np.ndarray) -> np.ndarray:
        if not self.ready:
            self.reset(target)
        self.inp.target_position = list(map(float, target))
        self.otg.update(self.inp, self.out)   # 0.19.x 的流式接口叫 update(不是 step)
        self.out.pass_to_input(self.inp)
        return np.asarray(self.out.new_position, dtype=np.float32)

# 相机名 → 数据集观测键(robot_config 用原生相机名做映射,直接同名)
CAMERA_TO_OBS_KEY = {
    "head_rgb": "observation.images.head_rgb",
    "left_wrist_rgb": "observation.images.left_wrist_rgb",
    "right_wrist_rgb": "observation.images.right_wrist_rgb",
}
# 腕部小窗标题(与遥操作数采时同款窗口)
WRIST_VIEWER_TITLES = {
    "left_wrist_rgb": "H1 left wrist RGB",
    "right_wrist_rgb": "H1 right wrist RGB",
}


def show_wrist_windows(images: dict) -> bool:
    """刷新左右腕两个小窗;返回 True 表示用户按了 q(提前结束本回合)。"""
    for camera_name, title in WRIST_VIEWER_TITLES.items():
        bgr = cv2.cvtColor(images[CAMERA_TO_OBS_KEY[camera_name]], cv2.COLOR_RGB2BGR)
        cv2.imshow(title, bgr)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def close_wrist_windows() -> None:
    for title in WRIST_VIEWER_TITLES.values():
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass  # 用户可能已手动关掉某个腕部窗口


# ==================== 实时推理监视窗 ====================
class InferenceMonitor:
    """cv2 画布实时展示推理结果:18 个关节各自一行 ——
    白线=state 历史,黄线=action 历史,右侧青色=最近一次前向的 50 步规划块。
    纯 numpy+cv2 绘制(不引入 matplotlib 交互,不拖慢 30Hz 主循环)。"""

    WIN = "LingBot Inference Monitor"
    ROW_H = 40          # 每关节行高
    LABEL_W = 70        # 左侧标签区宽
    HIST_N = 600        # 历史帧数(约 20s @30Hz)
    PLAN_W = 170        # 右侧规划块区宽(50 步,横向放大便于观察)

    # 展示的 18 维(数据集 20 维去掉腰部 [0:2])
    DIM_NAMES = (
        [j.replace("Left_", "L").replace("Right_", "R").replace("waist_", "W")
         .replace("_J", "J").replace("head_j", "Hj").replace("Head_j", "Hj")
         for j in UPPER_BODY_JOINTS[2:]]
        + ["Grip.L", "Grip.R"]
    )

    def __init__(self, task: str):
        self.task = task
        self.hist_state = np.full((self.HIST_N, 18), np.nan)
        self.hist_action = np.full((self.HIST_N, 18), np.nan)
        self.plan: np.ndarray | None = None      # 最近一次前向的 (chunk, 18) 规划
        self.frame = 0
        self.n_forward = 0
        self.last_infer_ms = 0.0
        self.n_rows = len(self.DIM_NAMES)
        self.W = self.LABEL_W + self.HIST_N + self.PLAN_W  # 70+600+170 = 840
        self.H = 36 + self.n_rows * self.ROW_H            # 顶栏 + 18 行
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)

    # ---- 每回合开始:清历史(窗口和顶栏累计保留) ----
    def clear(self) -> None:
        self.hist_state[:] = np.nan
        self.hist_action[:] = np.nan
        self.plan = None

    @staticmethod
    def _to20(v) -> np.ndarray:
        """state/action/plan 统一成 20 维数据集口径(18 维时腰部补 0),再取 [2:20]。"""
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] == 18:
            v = np.concatenate([np.zeros(2, np.float32), v])
        return v

    # ---- 数据更新(did_forward 由 run_episode 按全局步数判定,不用猜) ----
    def update(self, state, action, plan, frame: int,
               infer_ms: float, did_forward: bool) -> None:
        self.hist_state[:-1] = self.hist_state[1:]
        self.hist_action[:-1] = self.hist_action[1:]
        self.hist_state[-1] = self._to20(state)[2:20]     # 去掉腰部两维
        self.hist_action[-1] = self._to20(action)[2:20]
        if plan is not None:
            self.plan = self._to20_plan(plan)
        self.frame = frame
        if did_forward:
            self.last_infer_ms = infer_ms
            self.n_forward += 1

    def _to20_plan(self, plan) -> np.ndarray:
        plan = np.asarray(plan, dtype=np.float32)
        if plan.ndim == 3:            # (B, chunk, dim) → (chunk, dim)
            plan = plan[0]
        rows = [self._to20(p)[2:20] for p in plan]
        return np.stack(rows)

    # ---- 绘制 ----
    def _polyline(self, canvas, xs, ys, color, thick=1):
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, color, thick, cv2.LINE_AA)

    def render(self, fps: float = 0.0) -> np.ndarray:
        canvas = np.full((self.H, self.W, 3), 28, dtype=np.uint8)
        # 顶栏
        info = (f"f{self.frame}  fwd:{self.n_forward}  {self.last_infer_ms:.0f}ms"
                + (f"  {fps:.1f}fps" if fps else "") + f"  | {self.task[:44]}")
        cv2.rectangle(canvas, (0, 0), (self.W, 34), (45, 45, 45), -1)
        cv2.putText(canvas, info, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, "plan(50)", (self.LABEL_W + self.HIST_N + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        hist_x0, hist_x1 = self.LABEL_W, self.LABEL_W + self.HIST_N - 1
        plan_x0 = self.LABEL_W + self.HIST_N + 6
        for r in range(self.n_rows):
            y_top = 36 + r * self.ROW_H
            y_bot = y_top + self.ROW_H - 2
            y_mid = (y_top + y_bot) // 2
            # 行背景 + 分隔线 + 标签
            if r % 2 == 0:  # 隔行提亮(int16 中转防 uint8 溢出)
                band = canvas[y_top:y_bot + 1, self.LABEL_W:]
                canvas[y_top:y_bot + 1, self.LABEL_W:] = np.clip(
                    band.astype(np.int16) + 6, 0, 255).astype(np.uint8)
            cv2.line(canvas, (0, y_bot), (self.W, y_bot), (60, 60, 60), 1)
            cv2.putText(canvas, self.DIM_NAMES[r], (4, y_mid + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
            # 该维的取值范围(历史+规划一起定标,曲线才可比)
            vals = np.concatenate([self.hist_state[:, r], self.hist_action[:, r],
                                   self.plan[:, r] if self.plan is not None else []])
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-6:
                lo, hi = lo - 0.05, hi + 0.05
            pad = (hi - lo) * 0.15
            lo, hi = lo - pad, hi + pad

            def to_y(v):
                return y_bot - (np.asarray(v) - lo) / (hi - lo) * (y_bot - y_top)

            xs = np.arange(hist_x0, hist_x1 + 1, dtype=float)
            # state 历史(白,2px 底层) / action 历史(黄,1px 上层):
            # 两者贴近时白边露出=模型在"跟着状态走";分开=模型在主动引领
            st, ac = self.hist_state[:, r], self.hist_action[:, r]
            m = np.isfinite(st)
            if m.any():
                self._polyline(canvas, xs[m], to_y(st[m]), (235, 235, 235), 2)
            m = np.isfinite(ac)
            if m.any():
                self._polyline(canvas, xs[m], to_y(ac[m]), (60, 220, 255), 1)
            # 规划块(青色粗线):从历史区右端画 50 步
            if self.plan is not None and np.isfinite(self.plan[:, r]).all():
                px = np.linspace(hist_x1, plan_x0 + self.PLAN_W - 10, len(self.plan))
                self._polyline(canvas, px, to_y(self.plan[:, r]), (255, 255, 60), 2)
        # 历史/规划分界竖线
        cv2.line(canvas, (hist_x1, 36), (hist_x1, self.H), (80, 80, 80), 1)
        return canvas

    # ---- 刷新(连同腕部小窗一起,单次 waitKey 服务所有窗口) ----
    def show(self, images: dict, fps: float = 0.0) -> bool:
        for camera_name, title in WRIST_VIEWER_TITLES.items():
            bgr = cv2.cvtColor(images[CAMERA_TO_OBS_KEY[camera_name]], cv2.COLOR_RGB2BGR)
            cv2.imshow(title, bgr)
        cv2.imshow(self.WIN, self.render(fps))
        return (cv2.waitKey(1) & 0xFF) == ord("q")


# ==================== MuJoCo 环境侧(复刻 recorder 口径) ====================
class H1RolloutEnv:
    """封装名称索引 + 观测渲染 + 动作执行 + 回合重置,全部复刻数采记录器的口径。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, spawn_mode: str = "home"):
        self.model = model
        self.data = data
        if spawn_mode not in ("home", "band", "v3"):
            raise ValueError(f"未知 spawn 模式: '{spawn_mode}'(home=keyframe 固定 / v3=老固定点 / band=随机带备用)")
        self.spawn_mode = spawn_mode

        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise ValueError("H1_scene.xml 缺少 'home' keyframe")
        self.home_id = home_id
        self.home_ctrl = model.key_ctrl[home_id].copy()

        # 与 recorder 相同的四组索引
        self.joint_qposadr = []
        self.joint_actuators = []
        for joint_name in UPPER_BODY_JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_position")
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"模型缺少关节/执行器: '{joint_name}'")
            self.joint_qposadr.append(int(model.jnt_qposadr[joint_id]))
            self.joint_actuators.append(actuator_id)
        self.gripper_actuators = []
        for actuator_name in GRIPPER_ACTUATOR_NAMES.values():  # left 在前,right 在后
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                raise ValueError(f"模型缺少夹爪执行器: '{actuator_name}'")
            self.gripper_actuators.append(actuator_id)
        self.camera_ids = {}
        for camera_name in CAMERA_NAMES:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"模型缺少相机: '{camera_name}'")
            self.camera_ids[camera_name] = camera_id
        self.cube_body_ids = {}
        for cube_name in CUBE_BODY_NAMES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
            if body_id < 0:
                raise ValueError(f"模型缺少方块 body: '{cube_name}'")
            self.cube_body_ids[cube_name] = body_id

        # 渲染器惰性创建:观看模式必须等 viewer 起来再建 GL 上下文(recorder 同款教训)
        self.renderers: dict[str, mujoco.Renderer] | None = None

    def ensure_renderers(self) -> None:
        if self.renderers is None:
            self.renderers = {
                name: mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
                for name in CAMERA_NAMES
            }

    def reset_episode(self, settle_seconds: float) -> None:
        """整场景回 home keyframe(+可选 v4 随机出生带) + 静置:方块落稳,ctrl 保持 home。"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        # 数采回合起始夹爪全开(v4 每回合 f0 = [1.0, 1.0],操作员开录前松开扳机),
        # 而 home keyframe 的夹爪 ctrl 是 0(闭合)—— 直接沉降会让初始 state 与
        # 腕部相机图像落在训练分布之外。沉降前对齐成张开。
        for actuator_id in self.gripper_actuators:
            self.data.ctrl[actuator_id] = 1.0
        if self.spawn_mode == "band":
            spawns = apply_cube_spawns(self.model, self.data)
            print(f"  [ep] 随机出生带(备用) → {spawns_str(spawns)}")
        elif self.spawn_mode == "v3":
            spawns = apply_fixed_spawns(self.model, self.data, "v3")
            print(f"  [ep] v3 老固定点 → {spawns_str(spawns)}")
        mujoco.mj_forward(self.model, self.data)
        settle_end = self.data.time + settle_seconds
        while self.data.time < settle_end - 1e-9:
            mujoco.mj_step(self.model, self.data)

    def get_state(self) -> np.ndarray:
        """20 维 state,与 recorder.record_frame 完全一致(夹爪用 actuator_length)。"""
        state = np.empty(20, dtype=np.float32)
        for i, address in enumerate(self.joint_qposadr):
            state[i] = self.data.qpos[address]
        for i, actuator_id in enumerate(self.gripper_actuators):
            state[18 + i] = self.data.actuator_length[actuator_id]
        return state

    def render_images(self) -> dict:
        """三相机 uint8 HWC 图像,键为数据集观测键(observation.images.*)。"""
        self.ensure_renderers()
        images = {}
        for camera_name, renderer in self.renderers.items():
            renderer.update_scene(self.data, camera=self.camera_ids[camera_name])
            images[CAMERA_TO_OBS_KEY[camera_name]] = renderer.render()
        return images

    def apply_action(self, action: np.ndarray, waist_ctrl: np.ndarray) -> None:
        """20 维动作写 ctrl。注意 joint_actuators 的 0/1 号是腰部执行器 ——
        腰部未参与训练映射,用 home 值;臂/头/夹爪直接取动作对应位。"""
        for i, actuator_id in enumerate(self.joint_actuators):   # i=0,1 是 waist_J1/J2
            self._write_ctrl(actuator_id, waist_ctrl[i] if i < 2 else action[i])
        for i, actuator_id in enumerate(self.gripper_actuators):
            self._write_ctrl(actuator_id, action[18 + i])

    def _write_ctrl(self, actuator_id: int, value: float) -> None:
        lo, hi = self.model.actuator_ctrlrange[actuator_id]
        self.data.ctrl[actuator_id] = np.clip(value, lo, hi)

    def cube_in_bin(self, cube_name: str) -> bool:
        pos = self.data.xpos[self.cube_body_ids[cube_name]]
        return (
            abs(pos[0] - BIN_CENTER_XY[0]) < BIN_XY_TOL
            and abs(pos[1] - BIN_CENTER_XY[1]) < BIN_XY_TOL
            and BIN_Z_RANGE[0] < pos[2] < BIN_Z_RANGE[1]
        )

    def cube_positions(self) -> dict:
        return {name: self.data.xpos[body_id].copy() for name, body_id in self.cube_body_ids.items()}


# ==================== 主循环 ====================
def run_episode(env: H1RolloutEnv, policy: LingbotVLAv2Server, task: str,
                max_steps: int, settle: float, live: bool, reinfer_every: int,
                viewer=None, monitor: "InferenceMonitor | None" = None,
                bias_comp: dict | None = None, action_gain: float = 1.0,
                lead_clip: float = 0.0, grip_log: list | None = None,
                smoother: "RuckigSmoother | None" = None) -> dict:
    env.reset_episode(settle)
    policy.reset(ROBO_NAME)  # 清空动作队列(global_step 归零),每回合从零开始
    if smoother is not None:
        smoother.reset(env.get_state())   # 平滑层从当前真实状态零速启动
    if monitor is not None:
        monitor.clear()      # 曲线历史清空,窗口沿用
    if viewer is not None and viewer.is_running():
        viewer.sync()
    run_episode._last_frame_t = None   # 帧率统计的基准
    run_episode._fps = 0.0

    frame_duration = 1.0 / FPS
    ever_in = {name: False for name in CUBE_BODY_NAMES}
    last_action = env.home_ctrl.copy()
    n_forward = 0
    arm_jumps = []   # 臂命令帧间 |Δ|(平滑度度量;数采口径 p99≈0.02 rad)

    print(f"  [ep] 开始闭环(最多 {max_steps} 帧 ≈ {max_steps / FPS:.0f}s;每 {reinfer_every} 帧重推理)")
    for step in range(max_steps):
        # ── 滑动窗口重推(可选):偏离默认 50 时每 reinfer_every 帧清一次队列 ──
        if reinfer_every < policy.use_length and step > 0 and step % reinfer_every == 0:
            policy.reset(ROBO_NAME)

        # 1) 观测:图像 + state 同一仿真时刻(与数采口径一致)
        images = env.render_images()
        # 监视窗模式由 monitor.show() 统一刷图收键(避免一帧两次 waitKey 抢按键)
        if live and monitor is None and show_wrist_windows(images):
            print(f"  [ep] 第 {step} 帧:腕部窗口按 q,提前结束本回合")
            break
        obs = {**images, "observation.state": env.get_state(), "task": task}

        # 调试开关:LINGBOT_ROLOUT_DEBUG=1 时打印动作/状态并存首帧渲染图
        _dbg = bool(os.environ.get("LINGBOT_ROLOUT_DEBUG"))
        if _dbg and step == 0:
            for obs_key, im in images.items():
                cv2.imwrite(f"/tmp/rollout_dbg_{obs_key.split('.')[-1]}.png",
                            cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                print(f"    [dbg] {obs_key} shape={im.shape} mean={im.mean():.1f}")

        # 2) 推理:chunk_ret=False → 每帧拿单步动作,内部每 use_length 帧才真正前向
        t_infer0 = time.perf_counter()
        preds = policy.infer(obs)
        infer_ms = (time.perf_counter() - t_infer0) * 1000.0
        action_np = np.asarray(preds["action"], dtype=np.float32).reshape(-1)  # 20 维数据集口径
        if action_np.shape[0] == 18:  # 防御:万一返回的是 18 维映射口径
            action_np = np.concatenate([env.home_ctrl[:2], action_np])
        if not np.all(np.isfinite(action_np)):  # NaN/Inf 保护:沿用上一步动作
            print(f"  [ep] 第 {step} 帧动作含非有限值,沿用上一步")
            action_np = last_action             # last_action 已含补偿/增益,不重复加
        else:
            if action_gain != 1.0:  # 幅值增益(确诊实验):a'=state+λ(action−state),放大命令位移
                st = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
                action_np = st + action_gain * (action_np - st)
            if bias_comp:      # 执行侧偏置补偿(确诊实验):扣掉实测系统偏置后再执行
                for dim, value in bias_comp.items():
                    action_np[dim] += value
            if lead_clip > 0.0:    # 命令超前限幅(确诊实验):|a−state| 逐维钳制,消 chunk 边界 teleport
                st_clip = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
                action_np = st_clip + np.clip(action_np - st_clip, -lead_clip, lead_clip)
            if smoother is not None:     # Ruckig 平滑层:增益/限幅之后的最终输出过在线轨迹生成
                action_np = smoother.step(action_np)
        arm_jumps.append(float(np.abs(action_np[2:16] - last_action[2:16]).max()))
        last_action = action_np
        did_forward = policy.global_step % policy.use_length == 1  # 本帧刚做过一次前向
        if did_forward:
            n_forward += 1
        if _dbg and step in (0, 1, 2, 25, 49, 50, 51, 100, 200, 299):
            st = obs["observation.state"]
            print(f"    [dbg] f{step:3d} | 左臂state={np.round(st[2:9], 2)}")
            print(f"    [dbg] f{step:3d} | 右臂state={np.round(st[9:16], 2)}")
            print(f"    [dbg] f{step:3d} | 左臂pred ={np.round(action_np[2:9], 2)} | "
                  f"右臂pred={np.round(action_np[9:16], 2)}")
            print(f"    [dbg] f{step:3d} | 头/爪pred={np.round(action_np[16:20], 2)} | "
                  f"mean|a-s|={np.abs(action_np - st).mean():.4f}")

        # 2.5) 监视窗喂点:state=推理所见,action=本帧执行,plan=最新一次前向的整块规划
        if monitor is not None:
            chunk = policy.last_action_chunk          # None 或 {'action': (B,chunk,dim)}
            plan = chunk["action"][0] if isinstance(chunk, dict) else chunk
            monitor.update(obs["observation.state"], action_np, plan, step,
                           infer_ms, did_forward)

        # 3) 执行 1/30s(时间基准推进,与 teleop 调度同模式)
        env.apply_action(action_np, env.home_ctrl[:2])
        target = env.data.time + frame_duration
        while env.data.time < target - 1e-9:
            mujoco.mj_step(env.model, env.data)

        # 4) 成功判定 + 显示
        for cube_name in CUBE_BODY_NAMES:
            if env.cube_in_bin(cube_name):
                ever_in[cube_name] = True
        # 末端(双侧指环中点)到各自方块的最小距离追踪:量化"够不够得到"
        ee_min = getattr(run_episode, "_ee_min", None)
        if ee_min is None:
            def _hand_grasp_pt(side: str) -> np.ndarray:
                ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                         f"{side}_omnipicker_hand_{k}_loop_Link")
                       for k in ("narrow", "wide")]
                return np.mean([env.data.xpos[i] for i in ids], axis=0)
            ee_min = {c: (9.9, -1, None) for c in CUBE_BODY_NAMES}   # (最小距离, 帧号, 偏移向量)
            run_episode._grasp_pt = _hand_grasp_pt
            run_episode._ee_min = ee_min
        d_frame = {}
        for cube_name, side in zip(CUBE_BODY_NAMES, ("left", "right")):
            off = run_episode._grasp_pt(side) - env.cube_positions()[cube_name]
            d = float(np.linalg.norm(off))
            d_frame[cube_name] = d
            if d < ee_min[cube_name][0]:
                ee_min[cube_name] = (d, step, off)
        if grip_log is not None:   # 爪相位诊断:本帧执行的爪命令 + 双侧指端距离
            grip_log.append((step, float(action_np[18]), float(action_np[19]),
                             d_frame["red_cube_left"], d_frame["red_cube_right"]))
        end_in = {name: env.cube_in_bin(name) for name in CUBE_BODY_NAMES}
        if all(end_in.values()):
            print(f"  [ep] 第 {step} 帧:双块均已入桶,回合成功 ✓")
            break

        if viewer is not None and viewer.is_running():
            viewer.sync()
        if live:  # 观看模式按 30Hz 实时节奏回放;无头评估全速跑
            now = time.monotonic()
            if run_episode._last_frame_t is not None:  # 实测帧率(指数平滑)
                inst = 1.0 / max(now - run_episode._last_frame_t, 1e-6)
                run_episode._fps = 0.9 * run_episode._fps + 0.1 * inst
            run_episode._last_frame_t = now
            if monitor is not None and monitor.show(images, run_episode._fps):
                print(f"  [ep] 第 {step} 帧:监视窗按 q,提前结束本回合")
                break
            wall_target = (step + 1) * frame_duration
            overrun = time.monotonic() - run_episode._wall_start - wall_target
            if overrun < 0:
                time.sleep(-overrun)

        if (step + 1) % 150 == 0:
            cubes = env.cube_positions()
            cube_str = "  ".join(
                f"{name}: x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}"
                for name, pos in cubes.items()
            )
            print(f"  [ep] {step + 1}/{max_steps} 帧 | 前向 {n_forward} 次 | {cube_str}")

    end_in = {name: env.cube_in_bin(name) for name in CUBE_BODY_NAMES}
    ee_min_result = dict(run_episode._ee_min)

    def _dir_lab(v) -> str:   # 偏移向量→人话(基座系:+x前 +y左 +z上)
        if v is None:
            return "无"
        fx, fy, fz = float(v[0]), float(v[1]), float(v[2])
        return (f"{'前' if fx >= 0 else '后'}{abs(fx):.2f} "
                f"{'左' if fy >= 0 else '右'}{abs(fy):.2f} "
                f"{'上' if fz >= 0 else '下'}{abs(fz):.2f}")

    ee_str = "  ".join(f"{c}: {d:.3f}m@f{f} 偏移[{_dir_lab(v)}]"
                       for c, (d, f, v) in ee_min_result.items())
    print(f"  [ep] 指端-方块最小距离: {ee_str}(偏移=指端相对方块,前/左/右/上为基座系)")
    if arm_jumps:
        aj = np.asarray(arm_jumps)
        print(f"  [ep] 臂命令帧间|Δ|: p50 {np.percentile(aj, 50):.3f} / p99 {np.percentile(aj, 99):.3f}"
              f" / max {aj.max():.3f} rad(数采口径 p99≈0.02;chunk 边界 teleport≈0.16-0.49)")
    run_episode._ee_min = None                    # 每回合重置
    return {
        "steps": step + 1,
        "ever_in": ever_in,
        "end_in": end_in,
        "success": all(end_in.values()),
        "partial": any(end_in.values()),
        "final_cubes": env.cube_positions(),
        "ee_min": ee_min_result,
    }


def main() -> None:
    os.chdir(str(PROJECT_ROOT))  # robot_config/norm_stats 都是相对项目根的路径
    checkpoint = Path(ARGS.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"找不到 checkpoint 目录: {checkpoint}")
    task = ARGS.task or DEFAULT_TASK
    mode = "无头评估(全速)" if ARGS.headless else "仿真窗口 + 腕部小窗(30Hz 实时)"
    print(f"[load] checkpoint : {checkpoint}")
    print(f"[load] task       : {task}")
    print(f"[load] 模式       : {mode} | 回合数: {ARGS.episodes} | 重推间隔: {ARGS.reinfer_every} 帧")

    # 1) LingBot 策略服务(与开环评估/官方 websocket 部署同一个类)
    policy = LingbotVLAv2Server(
        str(checkpoint),
        robot_norm_path=ARGS.norm_path,   # ★与训练同一份统计表
        use_length=50,                    # chunk 长度(=训练 chunk_size)
        chunk_ret=False,                  # 每帧单步动作,内部整块缓存
        use_bf16=True,                    # 与训练同精度(12GB 显存)
        use_fp32=False,
        use_compile=False,                # 首跑不编译,快起稳跑
    )
    policy.reset(ROBO_NAME)
    smoother = None
    if ARGS.smooth == "ruckig":
        smoother = RuckigSmoother()
        print("[load] Ruckig 平滑层 : ON(臂 v2.0/a20/j200,爪 v4.0/a40/j400;"
              "限值=数采 p99×2-3,2026-09-15 实测)")
    bias_comp = BIAS_COMP_PRESETS.get(ARGS.bias_comp)   # off → None,不加
    if bias_comp:
        comp_str = ", ".join(f"dim{d}{v:+.3f}" for d, v in sorted(bias_comp.items()))
        print(f"[load] 偏置补偿 : ON ({ARGS.bias_comp}) → {comp_str}")
    if ARGS.action_gain != 1.0:
        print(f"[load] 幅值增益 : ON (λ={ARGS.action_gain:.2f}) → a'=state+λ(action−state)")
    print(f"[load] 动作键: {policy.action_key} | use_length: {policy.use_length}")

    # 2) MuJoCo 场景
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    # 头灯对齐数采:XRoboToolkit 的 MujocoTeleopController 启动时把 headlight
    # 调亮(ambient .4 / diffuse .8 / specular .6,见其 _robot_setup),数采视频
    # 全在这个光照下录制(mean≈116.6);xml 默认头灯只有 ≈82.5,直接 rollout 会
    # 造成训练/推理图像域差距(v4 0/10 的根因)。rollout 必须复刻同样的光照。
    model.vis.headlight.ambient = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
    model.vis.headlight.specular = [0.6, 0.6, 0.6]
    data = mujoco.MjData(model)
    env = H1RolloutEnv(model, data, spawn_mode=ARGS.spawn)
    if ARGS.spawn == "band":
        print("[load] 方块出生位 : 随机带(备用,已弃用)")
    elif ARGS.spawn == "v3":
        print("[load] 方块出生位 : v3 老固定点 (0.65,±0.30) — 新老对比评测用")
    else:
        print("[load] 方块出生位 : home keyframe(=v3 老固定点 0.65,±0.30,与数采 A 键同源)")
    print(f"[load] 场景: {MJCF_PATH.name} (nq={model.nq}, nu={model.nu}, timestep={model.opt.timestep})")

    # 3) 逐回合闭环(观看模式:主窗口相机固定为头部第一人称,同遥操作)
    results = []
    viewer_ctx = None
    monitor = None
    try:
        if not ARGS.headless:
            import mujoco.viewer as mujoco_viewer

            viewer_ctx = mujoco_viewer.launch_passive(model, data)
            viewer_ctx.__enter__()
            viewer_ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer_ctx.cam.fixedcamid = env.camera_ids["head_rgb"]
            env.ensure_renderers()  # viewer 起来之后再建离屏渲染器(GL 上下文顺序)
            for title in WRIST_VIEWER_TITLES.values():  # 腕部小窗随 viewer 一起建好
                cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            if not ARGS.no_monitor:  # 推理监视窗:曲线看模型"在想什么"
                monitor = InferenceMonitor(task)
                print("[view] 推理监视窗已启动:白=state 黄=action 青=50步规划块;按 q 跳过本回合")
            print("[view] MuJoCo 窗口(头部视角)+ 左右腕小窗已启动;Ctrl+C 退出")
        for ep in range(ARGS.episodes):
            print(f"\n===== 回合 {ep + 1}/{ARGS.episodes} =====")
            run_episode._wall_start = time.monotonic()
            grip_rows = [] if ARGS.grip_log else None
            result = run_episode(
                env, policy, task,
                max_steps=ARGS.max_steps, settle=ARGS.settle,
                live=not ARGS.headless, reinfer_every=ARGS.reinfer_every,
                viewer=viewer_ctx, monitor=monitor, bias_comp=bias_comp,
                action_gain=ARGS.action_gain, lead_clip=ARGS.lead_clip,
                grip_log=grip_rows, smoother=smoother,
            )
            if ARGS.grip_log and grip_rows:   # 爪相位诊断 CSV(追加,含回合号)
                new_file = not Path(ARGS.grip_log).exists()
                with open(ARGS.grip_log, "a", newline="") as f:
                    writer = csv.writer(f)
                    if new_file:
                        writer.writerow(["ep", "frame", "grip_cmd_L", "grip_cmd_R",
                                         "dist_L", "dist_R"])
                    writer.writerows([[ep + 1, *row] for row in grip_rows])
            results.append(result)
            cubes = result["final_cubes"]
            cube_str = "  ".join(
                f"{name}: ({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})" for name, pos in cubes.items()
            )
            print(
                f"  [ep] 结束:{result['steps']} 帧 | 结束入桶 L={'✓' if result['end_in']['red_cube_left'] else '✗'} "
                f"R={'✓' if result['end_in']['red_cube_right'] else '✗'} | "
                f"全程曾入桶 L={'✓' if result['ever_in']['red_cube_left'] else '✗'} "
                f"R={'✓' if result['ever_in']['red_cube_right'] else '✗'}\n"
                f"  [ep] 末态方块 {cube_str}"
            )
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
        if monitor is not None:
            try:
                cv2.destroyWindow(monitor.WIN)
            except cv2.error:
                pass
        if not ARGS.headless:
            close_wrist_windows()

    # 4) 汇总
    n = len(results)
    n_success = sum(r["success"] for r in results)
    n_partial = sum(r["partial"] for r in results)
    n_ever = sum(all(r["ever_in"].values()) for r in results)
    print("\n===== 成功率汇总 =====")
    print(f"回合数 {n} | 完全成功(双块入桶@结束) {n_success}/{n} = {n_success / n:.0%}")
    print(f"        | 部分成功(至少一块@结束) {n_partial}/{n} = {n_partial / n:.0%}")
    print(f"        | 全程曾双块入桶 {n_ever}/{n} = {n_ever / n:.0%}")
    print("提示:完全为 0 且部分成功 > 0,通常意味着单臂学会了、另一臂没学会,可回看末态方块坐标定位。")


if __name__ == "__main__":
    main()


# =====================================================================
# 【等价的 CLI 命令】(效果与本脚本完全相同)
#   conda activate lingbotv2
#   cd /home/mjq/robot_item/lingbot-vla-v2-main
#
#   # ① 观看模式:实时看推理(仿真窗口+腕部小窗+推理监视窗,30Hz,1 回合)
#   python mjq_lingbotvla_v2/Lingbot_H1_Rollout.py
#   #    加 --no-monitor 只要仿真画面(监视窗暂时不想要时)
#
#   # ② 无头批量:统计成功率(10 回合,全速)
#   python mjq_lingbotvla_v2/Lingbot_H1_Rollout.py --headless --episodes 10
#
#   # ③ 换 checkpoint 对比(开环并列的 20000 档)
#   python mjq_lingbotvla_v2/Lingbot_H1_Rollout.py --headless --episodes 10 \
#       --checkpoint output/h1/checkpoints/global_step_20000/hf_ckpt
#
#   # ④ 实验项:每 25 帧重推理(滑动窗口,可能提成功率但偏离训练口径)
#   python mjq_lingbotvla_v2/Lingbot_H1_Rollout.py --headless --episodes 10 --reinfer-every 25
#
#   # ⑤ 确诊实验:执行侧偏置补偿(右臂 R_J1−0.142/R_J4+0.188;all=左右都补)
#   python mjq_lingbotvla_v2/Lingbot_H1_Rollout.py --headless --episodes 10 --bias-comp right
#
# 【脚本内部等价于】(策略侧,与开环评估 load_policy_server 殊途同归)
#   from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
#   policy = LingbotVLAv2Server(model_path, robot_norm_path=..., use_length=50,
#                               chunk_ret=False, use_bf16=True, use_fp32=False)
#   policy.reset("h1")                    # 读 configs/robot_configs/h1.yaml + norm_stats
#   action = policy.infer(obs)            # obs=数据集口径字典;action=20 维数据集口径
# =====================================================================
