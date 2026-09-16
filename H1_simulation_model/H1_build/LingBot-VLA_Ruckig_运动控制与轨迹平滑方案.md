# LingBot-VLA + Ruckig 运动控制与轨迹平滑方案

## 1. 方案目标

本方案面向具身智能 VLA 机器人系统，重点解决：

- VLA 推理输出频率较低，与机器人底层高频控制存在时间尺度不匹配；
- Action Chunk / Temporal Ensemble 后仍存在动作突变；
- Chunk 边界处速度、加速度不连续；
- 直接发送 VLA 输出造成“一抽一抽”、抖动或运动不自然；
- 缺少统一的 VLA → 机器人运动控制接口。

核心目标：

> 将 VLA 输出的低频、离散动作目标，转换为满足位置、速度、加速度和 Jerk 约束的高频、连续、可执行机器人轨迹。

---

## 2. 推荐总体架构

```text
Camera / Observation
        ↓
      VLA
        ↓
   Raw Action Chunk
        ↓
 Temporal Ensemble
        ↓
 Action Validator
        ↓
   Ruckig OTG
        ↓
High-frequency Trajectory
        ↓
Robot Controller
        ↓
   Joint Driver
        ↓
      Motor
```

各层职责：

| 层级 | 核心职责 |
|---|---|
| VLA | 理解任务并生成动作意图 |
| Action Chunk | 提供连续动作序列 |
| Temporal Ensemble | 融合多次 VLA 推理结果 |
| Action Validator | 检查异常跳变、NaN、关节限位等 |
| Ruckig | 在线生成满足运动约束的平滑轨迹 |
| Robot Controller | 高频跟踪轨迹 |
| Joint Driver | 执行底层电机控制 |

---

## 3. 为什么不能直接发送 VLA 输出？

假设 VLA 每 100 ms 输出一次：

```text
t=0.0s    q=20°
t=0.1s    q=30°
t=0.2s    q=45°
t=0.3s    q=60°
```

直接执行可能产生较大的瞬时速度和加速度变化。

VLA 输出通常应该理解为：

> 期望动作目标 / Joint Position / Joint Delta

而不是“电机立即执行的控制量”。

因此需要在 VLA 与底层控制器之间加入轨迹生成层。

---

## 4. Ruckig 是什么？

Ruckig 是开源的实时 Online Trajectory Generator（OTG，在线轨迹生成器）。

它与普通线性插值的区别是：

```text
Current Position
Current Velocity
Current Acceleration
        +
Target Position
Target Velocity
Target Acceleration
        +
Velocity Limit
Acceleration Limit
Jerk Limit
        ↓
      Ruckig
        ↓
Time-parameterized trajectory
```

核心能力：

- 实时在线计算；
- 支持位置、速度、加速度状态；
- 支持最大速度约束；
- 支持最大加速度约束；
- 支持最大 Jerk 约束；
- 可以在运动过程中更新目标；
- 适合机器人实时控制循环。

官方项目：

https://github.com/pantor/ruckig

---

## 5. Ruckig 在 VLA 系统中的定位

不要把 Ruckig 理解成“修复 VLA 模型错误”的工具。

更准确地说：

> Ruckig 是 VLA 与机器人实时运动控制之间的运动学/轨迹接口层。

可以理解为：

```text
VLA
“我要去哪儿？”
        ↓
Ruckig
“怎样平滑、安全、满足约束地去？”
        ↓
Controller
“电机现在应该怎么动？”
```

---

## 6. 完整数据流

### 6.1 Raw Action Chunk

例如 6 自由度机械臂：

```text
A0 = [q1, q2, q3, q4, q5, q6]
A1 = [q1, q2, q3, q4, q5, q6]
A2 = [q1, q2, q3, q4, q5, q6]
...
```

### 6.2 Temporal Ensemble

多次 VLA 推理：

```text
Prediction 1 → Action A
Prediction 2 → Action B
Prediction 3 → Action C
        ↓
    加权融合
        ↓
   Fused Action
```

Temporal Ensemble 主要解决：

> 多次 VLA 推理结果不一致。

### 6.3 Action Validator

进入 Ruckig 前建议检查：

1. NaN / Inf
2. Joint Position Limit
3. Action Jump
4. Maximum Position Delta
5. Velocity Limit
6. 时间戳异常
7. Action Chunk 是否为空

异常时可以：

```text
Reject
  ↓
保持上一目标 / 安全停止 / 降级策略
```

---

## 7. Ruckig 的核心输入与输出

核心输入包括：

```text
current_position
current_velocity
current_acceleration

target_position
target_velocity
target_acceleration

max_velocity
max_acceleration
max_jerk
```

Ruckig 输出的是轨迹状态，例如：

```text
timestamp
position
velocity
acceleration
```

因此可以得到：

```text
t0   q0   dq0   ddq0
t1   q1   dq1   ddq1
t2   q2   dq2   ddq2
...
```

如果控制周期为 2 ms，则对应：

```text
500 Hz
```

可以形成：

```text
VLA
10~30 Hz
 ↓
Ruckig
500 Hz
 ↓
Robot Controller
500~1000 Hz
```

---

## 8. 为什么 Jerk 限制很重要？

定义：

```text
Position = q
Velocity = dq/dt
Acceleration = d²q/dt²
Jerk = d³q/dt³
```

即使位置连续，如果加速度突然变化，Jerk 也可能非常大。

可能造成：

- 突然顿挫；
- 机械臂“抽一下”；
- 电机声音变化；
- 末端振动；
- 轨迹不自然。

因此：

> 对 VLA 机器人执行而言，限制 Jerk 往往比单纯做位置低通滤波更合理。

---

## 9. 为什么不建议只使用低通滤波？

```text
VLA Action
     ↓
Low-pass Filter
     ↓
Robot
```

虽然可以减少高频噪声，但存在：

- 延迟；
- 无法天然保证速度、加速度、Jerk 约束；
- 可能改变 VLA 动作意图。

推荐：

```text
Temporal Ensemble
      ↓
轻量异常过滤
      ↓
Ruckig
```

---

## 10. Ruckig 与 MoveIt 2

如果使用 ROS 2 Humble，可以考虑：

```text
VLA
 ↓
ROS 2
 ↓
MoveIt 2
 ↓
Trajectory Processing
 ↓
Ruckig
 ↓
Robot Controller
```

MoveIt 2 提供完整机器人运动规划能力，也可以结合 Ruckig 做 jerk-limited trajectory smoothing。

MoveIt 2：

https://github.com/moveit/moveit2

Ruckig：

https://github.com/pantor/ruckig

如果 VLA 已经直接输出关节空间 Action，而目标只是在线平滑执行，则不一定需要完整 MoveIt 2，可以直接使用 Ruckig。

---

## 11. Ruckig 与 TOPPRA

TOPPRA：

https://github.com/hungpham2511/toppra

两者定位不同：

| 项目 | 主要作用 |
|---|---|
| Ruckig | 在线轨迹生成、实时平滑、Jerk 限制 |
| TOPPRA | 路径时间参数化、速度/加速度等约束 |
| MoveIt 2 | 完整机器人运动规划框架 |

对于 VLA Action Chunk 在线执行：

> 优先 Ruckig。

如果以后已有完整几何路径，需要进一步进行时间参数化：

> 再考虑 TOPPRA。

---

## 12. LingBot-VLA + SO101 推荐架构

```text
                         Camera
                           ↓
                    LingBot-VLA
                           ↓
                    Raw Action Chunk
                           ↓
                  Temporal Ensemble
                           ↓
                 ┌──────────────────┐
                 │ Action Validator │
                 │                  │
                 │ NaN/Inf Check    │
                 │ Joint Limit      │
                 │ Jump Detection   │
                 │ Delta Limit      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │     Ruckig       │
                 │                  │
                 │ Position         │
                 │ Velocity         │
                 │ Acceleration     │
                 │ Jerk             │
                 └────────┬─────────┘
                          ↓
                  100~500 Hz
                 Smooth Trajectory
                          ↓
                    SO101 Controller
                          ↓
                         Motor
```

---

## 13. VLA 与 Ruckig 的异步设计

推荐：

```text
VLA Thread
10~30 Hz
     │
     ↓
Target Buffer
     │
     ↓
Ruckig Thread
100~500 Hz
     │
     ↓
Robot Controller
500~1000 Hz
```

VLA 只负责不断更新目标，Ruckig 根据最新目标持续生成轨迹。

新目标到来时：

```text
当前运动
   ↓
New Target
   ↓
Ruckig Online Replanning
   ↓
连续的新轨迹
```

避免：

```text
停止
 ↓
跳到新目标
```

---

## 14. Action Chunk 与 Ruckig

假设：

```text
Chunk 1:
A1 A2 A3 A4 A5

Chunk 2:
B1 B2 B3 B4 B5
```

推荐：

```text
Chunk 1
 ↓
Temporal Ensemble
 ↓
Fused Action
 ↓
Ruckig
 ↓
Trajectory

Chunk 2
 ↓
Temporal Ensemble
 ↓
Fused Action
 ↓
Ruckig
 ↓
Online Replanning
```

不要简单地等待 Chunk 1 完整执行后再开始 Chunk 2。

---

## 15. 抖动问题的三层定位

### 15.1 模型层抖动

```text
VLA
 ↓
A1
A2
A1
A2
```

属于 Model-level jitter。

### 15.2 Chunk Boundary 抖动

```text
Chunk 1:
A1 A2 A3 A4 A5

Chunk 2:
          B1 B2 B3 B4 B5

A5 ≠ B1
```

属于 Action Chunk Boundary Discontinuity。

### 15.3 执行层抖动

即使：

```text
VLA → A1 A2 A3 A4
```

已经平滑，经过：

```text
Trajectory
 ↓
Controller
 ↓
Driver
 ↓
Motor
```

后仍然抖动，则应检查执行层。

---

## 16. 推荐日志

至少记录：

```text
timestamp

raw_vla_action
ensemble_action

ruckig_position
ruckig_velocity
ruckig_acceleration

actual_joint_position
actual_joint_velocity
```

并计算：

```text
Position Δ
Velocity
Acceleration
Jerk
```

重点观察：

1. Position Jump
2. Velocity Peak
3. Acceleration Peak
4. Jerk Peak

---

## 17. A/B 实验设计

### 实验 A

```text
VLA
 ↓
Temporal Ensemble
 ↓
Robot
```

### 实验 B

```text
VLA
 ↓
Temporal Ensemble
 ↓
Ruckig
 ↓
Robot
```

比较：

- Action Smoothness
- Position Jump
- Velocity Peak
- Acceleration Peak
- Jerk Peak
- Task Success Rate
- Execution Latency

推荐进一步增加：

| 实验 | Temporal Ensemble | Ruckig | 目的 |
|---|---:|---:|---|
| A | × | × | 原始基线 |
| B | ✓ | × | 验证 Ensemble |
| C | × | ✓ | 验证 Ruckig 独立效果 |
| D | ✓ | ✓ | 最终方案 |
| E | ✓ | ✓ + 严格 Jerk | 极限平滑测试 |

---

## 18. 参数调节建议

Ruckig 最重要的三个参数：

```text
max_velocity
max_acceleration
max_jerk
```

建议逐步调节：

### 第一阶段

固定 Velocity / Acceleration，逐渐降低 Jerk，观察动作。

### 第二阶段

逐步提高 Jerk，寻找平滑性与响应速度的平衡。

### 第三阶段

结合 SO101 实际硬件规格和实验测量确定最终参数。

原则：

> 不要凭感觉随意设置机器人运动极限，应以硬件规格和实际测试为依据。

---

## 19. 一个重要工程原则

> Ruckig 负责满足运动约束，而不是纠正 VLA 的任务决策。

例如 VLA 判断抓取方向错误：

```text
VLA
 ↓
错误动作
 ↓
Ruckig
 ↓
平滑地执行错误动作
```

所以：

```text
VLA
= 决策

Planner
= 路径/动作选择

Ruckig
= 轨迹生成

Controller
= 跟踪控制
```

职责必须分开。

---

## 20. 推荐的软件模块化

建议在 LingBot-VLA 项目中独立建立：

```text
vla_motion_adapter/
├── action_buffer.py
├── temporal_ensemble.py
├── action_validator.py
├── ruckig_trajectory.py
├── trajectory_logger.py
├── safety_limits.py
└── config.yaml
```

模块职责：

- `action_buffer.py`：管理 VLA Action Chunk；
- `temporal_ensemble.py`：融合多次推理；
- `action_validator.py`：异常检测；
- `ruckig_trajectory.py`：Ruckig 封装；
- `trajectory_logger.py`：记录 Raw / Fused / Ruckig / Actual；
- `safety_limits.py`：统一管理位置、速度、加速度、Jerk 限制。

---

## 21. 最终工程架构

```text
┌─────────────────────────────────────┐
│              AI Layer               │
│                                     │
│ Camera → LingBot-VLA → Action Chunk │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│          Action Processing           │
│                                     │
│ Temporal Ensemble                   │
│ Action Buffer                       │
│ Action Validator                    │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│          Motion Generation           │
│                                     │
│ Ruckig Online Trajectory Generator  │
│ Velocity / Acceleration / Jerk Limit│
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│          Real-time Control           │
│                                     │
│ Position / Velocity / Torque Loop  │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│              Robot                  │
│                                     │
│ SO101 / Joint Driver / Motor       │
└─────────────────────────────────────┘
```

---

## 22. 最终结论

对于当前 LingBot-VLA + SO101 场景，推荐：

> **LingBot-VLA → Action Chunk → Temporal Ensemble → Action Validator → Ruckig → Robot Controller**

其中：

- VLA：决定动作意图；
- Temporal Ensemble：解决多次推理不一致；
- Action Validator：拦截明显异常动作；
- Ruckig：把离散目标转换成连续、受速度/加速度/Jerk 约束的轨迹；
- Controller：高频跟踪轨迹；
- Motor：最终执行。

最重要的是：

> **Ruckig 不是简单的线性插值器，而是实时在线轨迹生成器。**

---

## 23. 推荐下一步实施路径

```text
现有 LingBot-VLA V7
        ↓
导出 Raw Action
        ↓
Temporal Ensemble
        ↓
接入 Ruckig
        ↓
离线生成平滑轨迹
        ↓
比较 Raw vs Ruckig
        ↓
确认抖动是否来自执行层
        ↓
再接入 SO101
```

这样能够把：

> 模型问题

和

> 运动执行问题

彻底分离。

如果 Ruckig 离线处理后的轨迹已经非常平滑，而真实机械臂仍然“一抽一抽”，则应重点检查：

```text
Ruckig → Controller → Driver → Motor
```

而不是继续修改 VLA。

---

## 24. 开源项目入口

- Ruckig：https://github.com/pantor/ruckig
- MoveIt 2：https://github.com/moveit/moveit2
- TOPPRA：https://github.com/hungpham2511/toppra
