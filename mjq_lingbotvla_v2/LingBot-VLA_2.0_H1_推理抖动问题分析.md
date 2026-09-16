# LingBot-VLA 2.0 H1 推理结果分析：机械臂“一抽一抽”抖动的原因

## 一、结论先说

从当前推理日志看，**“机器人一抽一抽地抖动”首先不应该简单归因于“12 万步没训练好”**。

目前最可疑的是：

> **50 帧 Action Chunk + 每 50 帧重新推理 + 单次推理约 0.35 秒 + Action Chunk 边界缺少平滑/重叠融合，导致动作在 Chunk 切换处出现明显跳变。**

其次需要排查：

1. 推理使用了 Eager Attention，而不是 Flash Attention 2；
2. 推理线程和控制线程是否同步阻塞；
3. Action Chunk 是否做了插值、平滑或 Temporal Ensemble；
4. 训练和推理时的 state/action normalization 是否完全一致；
5. H1 的 action mapping、joint 顺序、符号、单位和范围是否一致；
6. 300 回合数据的动作质量和一致性；
7. 最后才是“12 万步是否不足”。

你的 RTX 5090 32GB **显存和算力本身不是主要问题**。

---

# 二、日志中最关键的现象：每 50 帧才重新推理

日志中明确写着：

```text
[load] 模式 : 仿真窗口 + 腕部小窗(30Hz 实时) | 回合数: 1 | 重推间隔: 50 帧
```

也就是说：

```text
控制频率 = 30 Hz

50 帧 / 30 Hz ≈ 1.67 秒
```

因此目前的控制逻辑很可能是：

```text
t = 0
│
├── VLA 推理
│   └── 预测 50 个动作
│
├── 执行 action[0]
├── 执行 action[1]
├── ...
├── 执行 action[49]
│
│       ≈ 1.67 秒
│
├── 再次 VLA 推理
│   └── 再预测 50 个动作
│
├── 执行新的 50 个动作
│
└── ...
```

如果相邻两个 Action Chunk 的边界动作不连续，例如：

```text
第一个 Chunk 最后一个动作：
joint1 = 0.42

第二个 Chunk 第一个动作：
joint1 = 0.55
```

那么机械臂就会出现：

```text
0.42
 ↓
0.55
```

这种突变。

最终视觉表现就是：

> **动一下 → 停/缓一下 → 再突然动一下 → 再停 → 再动。**

这与目前观察到的“一抽一抽”高度吻合。

---

# 三、你的单次推理时间也明显偏高

日志显示：

```text
Denoise 10 steps
sample_actions batch=1 cost 0.35 s
```

也就是一次推理大约：

```text
350 ms
```

而官方 LingBot-VLA 2.0 README 给出的参考是：

> NVIDIA GeForce RTX 4090D 上，一次 inference、10 denoising steps，大约 130 ms。

因此你当前：

```text
RTX 5090
10 denoise
≈350 ms
```

明显偏慢。

这并不说明 5090 不够快，反而说明：

> **当前推理环境没有充分利用 GPU。**

---

# 四、日志明确显示：当前使用的是 Eager Attention

日志中有：

```text
You are attempting to use Flash Attention 2 without specifying a torch dtype.
This might lead to unexpected behaviour
```

随后又出现：

```text
=====Using Eager Attn=====
```

这意味着当前模型实际运行的是：

```text
Eager Attention
```

而不是：

```text
Flash Attention 2
```

这很可能是造成你单次推理达到约 350 ms 的重要原因之一。

理想情况下，应进一步检查：

```python
torch_dtype
```

以及：

```text
Flash Attention 2
```

是否正确安装并启用。

目标是让日志不再是：

```text
=====Using Eager Attn=====
```

而是实际使用 Flash Attention 2。

如果能够把：

```text
350 ms
```

降低到：

```text
100~150 ms
```

实时闭环控制能力会明显改善。

官方 README 给出的参考就是 4090D + 10 denoise ≈ 130 ms。

---

# 五、你的推理次数与日志完全对应

整个回合：

```text
1800 帧
```

每：

```text
50 帧
```

重新推理一次。

因此：

```text
1800 / 50 = 36 次
```

日志最后也显示：

```text
[ep] 1800/1800 帧 | 前向 36 次
```

这说明当前 rollout 的实际结构已经可以确定：

```text
1800 帧
÷
50 帧/次
=
36 次 VLA inference
```

即：

> **不是 VLA 连续地每 30Hz 做一次决策，而是大约每 1.67 秒重新生成一次 Action Chunk。**

这就是目前最需要关注的地方之一。

---

# 六、Action Chunk 不应该简单理解成“预测 50 步，然后硬执行 50 步”

你现在：

```text
use_length = 50
```

官方示例确实使用过：

```bash
--use_length 50
```

但需要注意：

> `use_length=50` 并不意味着实际机器人控制时必须采用“预测 50 步 → 完整执行 50 步 → 再预测 50 步”的方式。

更加适合实时闭环控制的结构通常是：

```text
预测 Action Chunk
        ↓
执行其中一部分
        ↓
重新获取视觉
        ↓
重新预测
        ↓
动作融合/平滑
        ↓
继续执行
```

而不是：

```text
预测 50
   ↓
硬执行 50
   ↓
重新预测 50
   ↓
硬执行 50
```

后者特别容易产生 Action Chunk 边界跳变。

---

# 七、不要简单地把 50 改成 5

这里有一个非常重要的细节。

如果直接改成：

```text
replan = 5 frames
```

而你的单次 VLA 推理仍然需要：

```text
350 ms
```

那么：

```text
30 Hz
↓
1 frame ≈ 33.3 ms
↓
5 frames ≈ 167 ms
↓
VLA 推理 ≈ 350 ms
```

就会出现：

```text
执行 5 帧
↓
等待推理 350 ms
↓
机器人卡顿
↓
新动作
↓
执行 5 帧
↓
等待推理
↓
卡顿
```

这样反而可能更加抖。

所以真正应该做的是：

# **异步推理 + Action Queue**

推荐结构：

```text
                 控制线程
                  30 Hz
                    │
                    ↓
          ┌──────────────────┐
          │   Action Queue   │
          └────────┬─────────┘
                   │
           连续执行动作
                   │
                   ↓
                H1 双臂


                 ↑
                 │ 异步更新
                 │
          ┌──────┴─────────┐
          │ LingBot-VLA 2.0│
          │                 │
          │ 当前图像        │
          │ 当前 state      │
          │                 │
          │ 10-step denoise │
          │                 │
          │ 输出 50-step    │
          └─────────────────┘
```

这样：

```text
控制循环
30Hz 持续运行

VLA
后台异步运行
```

两者互不阻塞。

---

# 八、第二个非常值得排查的问题：Action Chunk 是否做平滑

假设第一次预测：

```text
A1 = [a1, a2, a3, ..., a50]
```

第二次预测：

```text
A2 = [b1, b2, b3, ..., b50]
```

如果直接：

```text
a50
 ↓
b1
```

那么两个动作之间可能产生很大的跳变。

更好的方法是进行：

```text
Action Chunk Overlap
+
Linear Interpolation
+
Temporal Smoothing
```

例如：

```text
a46
a47
a48
a49
a50
   \
    \ 插值 / 融合
     \
      b1
      b2
      b3
```

甚至可以使用 Temporal Ensemble：

```text
多个历史 Action Chunk
        ↓
按照相同时间位置对齐
        ↓
加权平均
        ↓
最终控制动作
```

这样可以明显降低机械臂动作的突变。

---

# 九、你的实验结果说明模型并不是完全没有学到

这个数据非常重要：

```text
red_cube_left:
最小距离 0.231 m @ f846

red_cube_right:
最小距离 0.063 m @ f552
```

也就是：

### 左臂

```text
23.1 cm
```

离方块还比较远。

### 右臂

```text
6.3 cm
```

已经非常接近方块。

因此不能简单判断：

> “模型完全没学会。”

更合理的解释是：

```text
模型具有一定的目标趋近能力
        ↓
右臂已经接近目标
        ↓
但是没有形成稳定完整的任务行为
        ↓
接近 → 对准 → 闭合夹爪 → 抬起 → 移动 → 放入
```

也就是说，目前更像：

> **模型学到了一部分 task prior，但是闭环控制质量、动作连续性或数据映射存在问题。**

---

# 十、训练数据量：300 回合不一定意味着数据一定够

你有：

```text
300 episodes
120,000 steps
```

如果平均每回合大约 400 steps：

```text
300 × 400 ≈ 120,000
```

那么确实大约就是 300 条完整示范。

对于：

> Pick up the red cube and put it into the green bin

这种相对简单的任务，300 回合理论上有可能学会。

但前提是：

> **300 条示范的数据质量足够高，而且动作模式具有足够的一致性。**

例如如果数据中：

```text
第1回：左手先抓
第2回：右手先抓
第3回：左手绕过去
第4回：右手绕过去
第5回：两手同时运动
```

那么模型容易学到：

> “看到红色方块以后，机械臂应该往某个大概方向运动。”

而不是：

> “看到红色方块后，以稳定的轨迹完成整个抓取和放置任务。”

因此：

**数据质量比单纯的数据条数更加重要。**

---

# 十一、必须重点检查 Action / State Mapping

LingBot-VLA 2.0 使用统一的 canonical action representation。

官方文档中定义了 55 维状态/动作表示，包括：

```text
14 维 arm joint position
14 维 end-effector pose
2 维 gripper
12 维 hand
4 维 waist
2 维 head
3 维 mobility
4 维 reserved
```

你的日志显示：

```text
[load] 动作键: ['action']
```

所以需要确认 H1 数据是否正确映射到了 LingBot-VLA 2.0 的 action space。

尤其要检查：

```text
左臂 joint 顺序
右臂 joint 顺序
左夹爪
右夹爪
head
waist
```

有没有：

```text
左右臂颠倒
joint 顺序错误
符号错误
角度/弧度错误
单位错误
动作范围错误
state/action 对不上
```

这些问题都可能导致：

> 模型看起来像“完全没学会”，实际上是动作映射错了。

---

# 十二、Normalization 是另一个高优先级检查项

LingBot-VLA 2.0 官方文档明确区分了 Real-World 和 RoboTwin 的归一化方式：

### Real-World

```text
per-joint meanstd
```

### RoboTwin

```text
bounds_99_woclip
```

官方配置文档明确说明，两类训练配置在：

```text
norm_type
loss_type
```

上存在差异。

你的模型是：

```text
H1
自己的 300 回合数据
```

因此必须确认：

```text
训练 normalization
=
推理 normalization
```

例如不能出现：

```text
训练：
meanstd

推理：
bounds_99_woclip
```

也不能出现：

```text
训练：
degrees

推理：
radians
```

否则模型输出会完全失真。

---

# 十三、RTX 5090 32GB 不是目前最值得怀疑的问题

你的硬件：

```text
RTX 5090
32 GB VRAM
```

对于当前模型来说显存完全够用。

现在更应该关注的是：

```text
GPU 有没有正确使用
        ↓
Attention 是否正确
        ↓
dtype 是否正确
        ↓
Flash Attention 是否启用
        ↓
推理线程是否阻塞控制线程
```

而不是继续单纯增加 GPU 算力。

---

# 十四、哪些 Warning 可以暂时忽略

## 1. Qt 字体 Warning

```text
QFontDatabase: Cannot find font directory
```

这主要是 OpenCV/Qt 字体路径问题。

一般不会导致：

```text
机械臂动作异常
模型输出错误
```

所以目前不用把它作为重点。

## 2. GLFW Warning

程序最后出现：

```text
GLFWError: (65537) b'The GLFW library is not initialized'
```

它出现在 episode 结束后：

```text
成功率汇总
```

之后。

因此它基本不是机械臂“一抽一抽”的核心原因。

---

# 十五、当前问题的优先级排序

建议按照下面的优先级排查：

| 优先级 | 问题 | 判断 |
|---|---|---|
| 🔴 1 | 50 步 Action Chunk 硬执行、Chunk 边界跳变 | 非常高 |
| 🔴 2 | Eager Attention 导致推理约 350 ms | 高 |
| 🔴 3 | 推理线程/控制线程同步阻塞 | 非常高 |
| 🔴 4 | Action Chunk 没有 Temporal Smoothing | 非常高 |
| 🟠 5 | Action/State normalization 不一致 | 高 |
| 🟠 6 | H1 action mapping / joint order 不一致 | 高 |
| 🟠 7 | 300 回合数据质量不足 | 中 |
| 🟡 8 | 12 万步不够 | 目前不能确定 |
| 🟢 9 | RTX 5090 算力不够 | 基本不是 |
| 🟢 10 | Qt/GLFW warning | 基本不是 |

---

# 十六、最关键的诊断实验

现在**不要马上重新训练**。

你当前 checkpoint：

```text
/home/mjq/robot_item/lingbot-vla-v2-main/output/h1_v4_w2/checkpoints/global_step_120000/hf_ckpt
```

完全可以继续用于诊断。

最重要的实验是：

## 实验 1：直接打印 VLA 输出的 50 步 Action

例如打印：

```text
step 0
step 1
step 2
...
step 49
```

以及：

```text
joint1
joint2
joint3
...
gripper
```

观察动作是否连续。

### 如果模型输出本身就是跳的

例如：

```text
0.21
0.22
0.24
0.23
0.25
0.62   ← 突变
0.24
...
```

那么应该重点检查：

```text
训练
normalization
action mapping
模型本身
```

### 如果模型输出非常平滑

例如：

```text
0.21
0.22
0.23
0.24
0.25
0.26
...
```

但是机械臂实际表现：

```text
动一下
↓
停
↓
动一下
↓
停
```

那么就应该重点检查：

> **rollout 的 Action Queue、Chunk 切换、推理阻塞和动作执行代码。**

---

# 十七、下一步最值得检查的代码

需要重点查看：

```python
sample_actions(...)
```

以及：

```python
use_length
```

```python
lead_clip
```

还有真正执行动作的代码，例如：

```python
for action in actions:
    ...
```

以及：

```text
每 50 帧重新推理
```

的具体实现。

重点检查以下 6 件事情：

```text
① VLA 到底多久推理一次

② 50 个 action 到底如何执行

③ VLA 推理期间 MuJoCo 仿真是否被阻塞

④ 相邻 Action Chunk 是否存在动作跳变

⑤ 是否有 Action Interpolation / Temporal Ensemble

⑥ normalization / action mapping 是否和训练完全一致
```

---

# 十八、推荐的最终闭环结构

对于你的 H1 双臂 30Hz 仿真，更推荐：

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
               │                    │
               │ 10-step denoise   │
               │ Action Chunk = 50  │
               └─────────┬──────────┘
                         │
                         ↓
               ┌────────────────────┐
               │ Action Queue        │
               │                    │
               │ Chunk Overlap       │
               │ Interpolation       │
               │ Temporal Smoothing  │
               └─────────┬──────────┘
                         │
                         ↓
                    30 Hz 控制
                         │
                         ↓
                    H1 双臂执行
                         │
                         └──────→ 新 State
                                      │
                                      └──→ 新视觉
```

核心思想是：

> **VLA 负责“想下一段动作”，控制器负责“连续、稳定地执行动作”。**

而不是让 VLA 的每一次重新推理直接决定机械臂下一瞬间的动作。

---

# 十九、最终判断

综合你这次的日志，我目前**不建议直接得出“12 万步训练失败”的结论**。

更合理的判断是：

```text
12万步 checkpoint
       ↓
模型能够输出一定程度的目标趋近行为
       ↓
右臂曾经距离方块只有 6.3cm
       ↓
说明模型并非完全没有学到
       ↓
但最终没有完成抓取和放置
       ↓
同时存在明显的 Action Chunk / 推理实时性问题
       ↓
还需要排查 normalization 和 action mapping
```

所以当前最应该做的是：

> **先检查 rollout，而不是马上重新训练。**

尤其是：

```text
50 帧重推理
+
350ms 单次推理
+
Eager Attention
+
Action Chunk 是否平滑
+
推理是否阻塞控制
```

这几个问题解决以后，再判断这 12 万步模型到底有没有真正学会。

如果确认 rollout 没问题，而模型输出本身仍然不正确，再考虑：

```text
12万步
↓
继续训练
```

或者重新检查：

```text
数据质量
normalization
robot config
action mapping
训练 loss
checkpoint
```

这样排查效率会高很多。
