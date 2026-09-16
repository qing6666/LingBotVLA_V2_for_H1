# LingBot-VLA 2.0 H1：模型输出与动作执行器问题排查方案

## 一、排查目标

当前最重要的事情不是继续训练，而是先确认：

> **到底是 LingBot-VLA 模型输出的 Action 本身有问题，还是模型输出正常，但 rollout / Action Executor 把动作执行坏了。**

当前已知现象：

- 训练：300 回合
- 训练步数：120,000 steps
- 推理 checkpoint：
  `/home/mjq/robot_item/lingbot-vla-v2-main/output/h1_v4_w2/checkpoints/global_step_120000/hf_ckpt`
- RTX 5090 32GB
- 控制频率：30 Hz
- Action Chunk：50
- 每 50 帧重新推理
- 单次 VLA 推理：约 0.35 s
- 当前使用：Eager Attention
- 右臂距离红色方块最近：约 6.3 cm
- 最终左右方块均未入绿箱
- 机器人表现为“一抽一抽”地运动

因此第一步应该进行**纯模型输出诊断**。

---

## 二、整体诊断思路

将整个系统拆成：

```text
                    LingBot-VLA
                         │
                         ↓
                 预测 50 步 Action
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
       保存/分析 Action          发送给 H1
              │                     │
              ↓                     ↓
       Action 本身是否平滑？     实际机器人是否平滑？
              │                     │
       ┌──────┴──────┐       ┌──────┴──────┐
       ↓             ↓       ↓             ↓
      是             否      是             否
       │             │       │             │
       ↓             ↓       ↓             ↓
  执行器/控制器     模型/归一化   执行器正常    执行器有问题
  重点排查          /mapping
```

核心原则：

> **先把模型输出和机器人执行解耦。**

---

## 三、第一阶段：完全不让机器人执行，只观察模型输出

找到：

```text
mjq_lingbotvla_v2/Lingbot_H1_Rollout.py
```

重点寻找模型推理调用，例如：

```python
sample_actions(...)
```

或者类似：

```python
actions = model(...)
actions = policy(...)
```

在模型输出 Action Chunk 之后，增加调试输出。

---

## 四、首先确认 Action 的 Tensor Shape

如果模型输出：

```text
[1, 50, N]
```

可以临时打印：

```python
print("========== ACTION CHUNK ==========")
print("shape:", actions.shape)

for i in range(min(50, actions.shape[1])):
    print(
        f"step {i}: "
        f"{actions[0, i].detach().cpu().numpy()}"
    )
```

如果实际输出：

```text
[50, N]
```

则：

```python
print("========== ACTION CHUNK ==========")
print("shape:", actions.shape)

for i in range(min(50, actions.shape[0])):
    print(
        f"step {i}: "
        f"{actions[i].detach().cpu().numpy()}"
    )
```

注意：以上只是调试模板，最终应根据实际代码中的变量名和 Tensor shape 修改。

---

## 五、建议保存 CSV，而不是长期打印

建议建立：

```text
debug_actions/
├── chunk_000.csv
├── chunk_001.csv
├── chunk_002.csv
├── ...
```

CSV 格式：

```text
step,joint1,joint2,joint3,joint4,...
0,...
1,...
2,...
...
49,...
```

这样可以进一步画曲线和进行数值分析。

---

## 六、检查单个 Action Chunk 内部是否平滑

正常示例：

```text
step 0   0.21
step 1   0.22
step 2   0.23
step 3   0.24
step 4   0.25
...
step 49  0.42
```

说明单个 Action Chunk 内部没有明显跳变。

异常示例：

```text
0.21
0.22
0.23
0.24
0.82  ← 突然跳变
0.25
0.26
```

说明模型输出本身存在异常跳变。

此时优先检查：

- normalization
- action mapping
- 数据质量
- checkpoint
- 推理预处理
- denoising
- Action Expert

---

## 七、最关键：检查 Chunk 与 Chunk 的交界处

例如：

### Chunk 0

```text
step 47 = 0.401
step 48 = 0.409
step 49 = 0.416
```

### Chunk 1

```text
step 0 = 0.421
step 1 = 0.428
step 2 = 0.435
```

这是连续的：

```text
0.401
 ↓
0.409
 ↓
0.416
 ↓
0.421
 ↓
0.428
```

如果是：

```text
Chunk 0 step 49 = 0.416
Chunk 1 step 0  = 0.612
```

那么：

```text
0.416
 ↓
0.612
```

出现明显跳变，即典型的 **Action Chunk Boundary Jump**。

---

## 八、检查 Action 与当前 Robot State 的差值

记录：

```text
current_state
action[0]
```

例如：

```text
current_state = 0.401
action[0] = 0.412

Δ = 0.011
```

比较合理。

如果：

```text
current_state = 0.401
action[0] = 0.712

Δ = 0.311
```

就需要重点检查：

- normalization
- action space
- 单位
- joint mapping
- action range
- state/action 对齐

---

## 九、第三种非常重要的情况

可能出现：

```text
Chunk 0

0.20
0.21
0.22
...
0.39
0.40

Chunk 1

0.58
0.59
0.60
...
```

每个 Chunk 内部都平滑，但 Chunk 之间跳变。

这说明：

> **模型不一定完全错误，可能是每次重新规划得到的 Action Chunk 不一致，而 rollout 直接把两个 Chunk 硬连接起来。**

此时应该考虑：

```text
Action Chunk Overlap
+
Linear Interpolation
+
Temporal Smoothing
+
Temporal Ensemble
```

---

## 十、不要简单地把重推理间隔从 50 改成 5

当前：

```text
控制频率 = 30 Hz
```

所以：

```text
1 frame ≈ 33.3 ms
```

如果：

```text
replan = 5 frames
```

则：

```text
5 × 33.3 ≈ 167 ms
```

而当前 VLA 推理：

```text
≈350 ms
```

会造成：

```text
执行 5 帧
 ↓
等待推理 350 ms
 ↓
机器人卡顿
 ↓
执行新动作
 ↓
再等待
```

所以不能简单粗暴地：

```text
50 → 5
```

---

## 十一、推荐架构：异步推理 + Action Queue

推荐结构：

```text
                 ┌─────────────────┐
                 │  H1 当前 State  │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │   当前 Camera   │
                 └────────┬────────┘
                          │
                          ↓
               ┌────────────────────┐
               │ LingBot-VLA 2.0    │
               │ 10-step denoise    │
               │ Action Chunk = 50  │
               └─────────┬──────────┘
                         │
                         ↓
               ┌────────────────────┐
               │ Action Queue       │
               │ Chunk Overlap      │
               │ Interpolation      │
               │ Temporal Smoothing │
               └─────────┬──────────┘
                         │
                         ↓
                    30 Hz 控制
                         │
                         ↓
                    H1 双臂执行
                         │
                         └────→ 新 State → 新视觉
```

核心思想：

> **VLA 负责预测未来动作，控制器负责稳定、连续地执行动作。**

---

## 十二、第四个实验：只执行 Action[0]

这是一个很有效的隔离实验。

逻辑：

```text
VLA
 ↓
50 actions
 ↓
只取 action[0]
 ↓
执行一次
 ↓
停止
```

连续重复几次并记录：

```text
current_state
action[0]
```

正常示例：

```text
第1次：
current = 0.40
action0 = 0.41

第2次：
current = 0.41
action0 = 0.42

第3次：
current = 0.42
action0 = 0.43
```

异常示例：

```text
第1次：
current = 0.40
action0 = 0.72

第2次：
current = 0.41
action0 = 0.31

第3次：
current = 0.42
action0 = 0.68
```

如果是后一种，模型输出本身就非常可疑。

---

## 十三、第五个实验：比较 Action 与 Actual State 曲线

最终记录两条曲线：

```text
Joint Position
 ^
 |                    Action
 |                  ╱
 |               ╱
 |            ╱
 |         ╱
 |      ╱
 |____╱________________ Time
      Actual State
```

如果：

```text
Action
```

非常平滑，而：

```text
Actual State
```

一抽一抽：

> **重点怀疑 Action Executor / 控制循环。**

如果：

```text
Action
```

本身就一抽一抽：

> **重点怀疑模型、数据、normalization 或 action mapping。**

---

## 十四、`lead-clip` 暂时不要修改

当前：

```bash
--lead-clip 0.05
```

第一轮诊断保持：

```bash
--lead-clip 0.05
```

不要同时修改：

```text
lead-clip
replan interval
use_length
normalization
```

否则无法判断到底是哪一个变量导致结果变化。

---

## 十五、推理速度也需要单独记录

当前：

```text
RTX 5090 32GB
Denoise 10 steps
≈350 ms
=====Using Eager Attn=====
```

需要检查：

```text
torch dtype
Flash Attention 2
CUDA
PyTorch
Transformers
模型加载配置
```

目标是确认是否真正启用了 Flash Attention 2。

---

## 十六、如果确认模型输出有问题，再检查训练侧

### 1. Normalization

确认：

```text
训练 normalization
=
推理 normalization
```

例如不能：

```text
训练：meanstd
推理：bounds_99_woclip
```

也不能：

```text
训练：degrees
推理：radians
```

### 2. Action Mapping

检查：

```text
左臂 joint 顺序
右臂 joint 顺序
左夹爪
右夹爪
head
waist
```

是否存在：

```text
左右臂颠倒
joint 顺序错误
符号错误
单位错误
范围错误
state/action 不对应
```

### 3. 数据质量

300 回合并不一定意味着数据一定足够。

重点检查：

```text
动作是否一致
抓取轨迹是否稳定
左右臂行为是否一致
成功示范比例
失败示范比例
异常动作
```

---

## 十七、当前几个 Warning 的优先级

### Qt 字体

```text
QFontDatabase: Cannot find font directory
```

暂时可以忽略，通常不会导致机械臂动作抖动。

### GLFW

```text
GLFWError: (65537)
The GLFW library is not initialized
```

当前出现在程序结束后，因此不是“一抽一抽”现象的主要原因。

---

## 十八、最终诊断树

```text
开始
  │
  ↓
保存 VLA 输出 Action Chunk
  │
  ↓
Action Chunk 内部是否平滑？
  │
  ├── 否 ──→ 模型/Normalization/Mapping/数据问题
  │
  └── 是
       │
       ↓
   Chunk 边界是否平滑？
       │
       ├── 否 ──→ Action Chunk 执行/融合问题
       │
       └── 是
            │
            ↓
       Action[0] 与当前 State
       是否合理接近？
            │
            ├── 否 ──→ Normalization/Mapping/模型问题
            │
            └── 是
                 │
                 ↓
        Action 与 Actual State
        曲线是否一致？
                 │
          ┌──────┴──────┐
          ↓             ↓
        不一致          一致
          │             │
          ↓             ↓
      Executor/      模型和执行器
      控制器问题       基本正常
```

---

## 十九、推荐实际操作顺序

### 第一步

不重新训练。

继续使用：

```text
global_step_120000/hf_ckpt
```

### 第二步

给：

```text
Lingbot_H1_Rollout.py
```

增加 Action Debug。

### 第三步

记录：

```text
每个 Chunk 的 50 步 Action
```

### 第四步

计算：

```text
Chunk 内最大跳变
Chunk 边界最大跳变
action[0] - current_state
每个 joint 的最大变化量
左右臂分别统计
gripper 单独统计
```

### 第五步

画：

```text
Joint Position vs Time
```

比较：

```text
Action
Actual State
```

### 第六步

根据结果二分：

**如果 Action 本身坏：**

```text
Normalization
Action Mapping
训练数据
推理预处理
Checkpoint
```

**如果 Action 好，但机器人执行坏：**

```text
Action Queue
Action Chunk
控制频率
推理阻塞
Interpolation
Temporal Ensemble
MuJoCo actuator
```

---

## 二十、最终结论

目前**不要继续训练**。

最有效的第一步是回答：

> **这个 120k checkpoint 到底输出了什么？**

只要把：

```text
VLA Action
```

和：

```text
H1 Actual State
```

同时记录下来，就能把问题从“看起来机器人在抖”变成**数据层面的明确诊断**。

最终建议做一个专门的：

```text
LingBot-VLA H1 Action Debugger
```

自动输出：

```text
debug_actions/
├── chunk_000.csv
├── chunk_001.csv
├── ...
├── action_state.csv
└── summary.csv
```

并计算：

```text
最大动作跳变
Chunk 边界跳变
State-Action 偏差
最大速度
最大加速度
左右臂分别统计
夹爪统计
```

这样就可以明确判断：

> **问题到底发生在 LingBot-VLA 模型，还是发生在 rollout / Action Executor。**
