# 从 URDF 到 VR 遥操作数采：全流程复现教程

> 以 H1 双臂人形 + 双 OmniPicker 夹爪 + PICO 4 Ultra + MuJoCo + LeRobotDataset v3.0
> 的实际复现为范例。目标读者：拿到一台**新机器人模型的 URDF**，要一路做到
> **VR 遥操作 → 数采 → 可训练的 v3.0 数据集**的人。
>
> 配套资料：`H1_仿真学习笔记.md`（踩坑细节全集）、`H1_build/scripts/01~12`
> （每步可运行脚本 + 自测）。本文是流程骨架，笔记是血肉。

---

## 0. 全景：整条链路长什么样

```
SolidWorks/厂商 URDF
   │ ① 体检 (结构/限位/mesh 路径)
   ▼
MuJoCo MJCF ──► ② 执行器 ──► ③ 末端夹爪 ──► ④ 碰撞体 ──► ⑤ 场景
   │                                                        │
   │                                        ⑥ IK 用纯运动学 URDF ◄─┘
   ▼
h1_pico_teleop.py ◄── PICO 头显/手柄 (WiFi → PC Service → SDK)
   │ ⑦ 遥操作 (clutch + 模拟量夹爪)
   ▼
⑧ --record 数采 (LeRobotDataset v3.0)
   │
   ▼
smolvla 训练
```

**核心工程原则**：每一步写成一个**编号脚本**（输入文件→输出文件→self_test
断言），可重复执行、可独立验证。出问题时能定位到具体某一步，而不是在一
个大脚本里大海捞针。

---

## 1. 拿到新模型 URDF：先体检，别急着转

### 1.1 检查清单

| 检查项 | 怎么查 | 见过的问题 |
|---|---|---|
| mesh 路径 | 打开 URDF 看 `<mesh filename>` | `package://` 前缀 MuJoCo 不认，要改成相对路径 |
| 关节限位 | 逐个 joint 看 `limit lower/upper` | SolidWorks 导出可能**全为 0**（等于所有关节焊死） |
| 命名一致性 | joint/link 名单打印 | 混杂大小写（如 `Right_j6` 小写 j）、左右不对称 |
| 惯性参数 | 每个 link 有 `<inertial>` 且 mass>0 | 缺失会编译失败或平衡惯量后仍不稳 |
| 关节类型 | revolute/continuous/fixed 分布 | fixed 连接会折叠 link，nq 对不上预期 |
| 单位/坐标系 | 目测模型尺寸 | STL 毫米/米混用、Z-up/其他 |

### 1.2 编译冒烟（两个引擎都过一遍）

```python
import mujoco
m = mujoco.MjModel.from_xml_path("model.urdf")   # MuJoCo 能直接读 URDF
print(m.nq, m.nv, m.njnt)                        # 和样本数对得上吗
# placo/pinocchio 也加载一遍 —— IK 链路靠它
import placo; robot = placo.RobotWrapper("model.urdf")
```

加载不了先修 URDF，**不要带病转换**。

---

## 2. 环境与依赖（两个 conda 环境，职责分离）

| 环境 | 用途 | 关键内容 |
|---|---|---|
| `lerobot312` (Python 3.12) | 建模/流水线/校验/训练 | mujoco、placo、lerobot（editable 安装）、pyarrow、ffmpeg |
| `xr-robotics` (Python 3.12) | PICO 遥操作 + 实机数采 | XRoboToolkit SDK（编译安装）、lerobot[dataset]、placo |

要点：
- **lerobot 要求 Python ≥3.12**；装 SDK 的脚本若默认建 3.10 环境，手动
  `conda create -n xr-robotics python=3.12` 再跑安装脚本。
- placo 及其 cmeel 依赖**必须版本锁定**（如 placo 0.9.16 配套套件），
  否则 libpinocchio ABI 错配；卸载过 coal-library 后要
  `--force-reinstall --no-deps coal-library` 恢复共享 prefix 里的模块。
- opencv **只装 GUI 版**（opencv-python），装了 headless 会抢共享 cv2
  目录，`cv2.namedWindow` 直接 not implemented。
- cmake ≥3.31 有兼容坑，需要时降级。
- 视频编码走 PyAV/torchcodec，ffmpeg 需在 PATH。

---

## 3. URDF → MJCF：逐步构建流水线

转换后**不要手改大 XML**，继续用脚本流水线（H1_build 的 01~08 即模板）：

| 步 | 做什么 | 关键参数/坑 |
|---|---|---|
| ① | URDF 体检 | 见上节 |
| ② | 转 MJCF | 编译验证 nq/nv/nbody；**URDF 的 dynamics/friction 会写死在元素上、压过 class 默认值** |
| ③ | 执行器 | 每关节一个 position actuator（`{joint}_position` 命名约定）；kp/kv 按电机型号分档；**forcerange 是速度的隐形旋钮**——限幅太小表现为"蠕变慢"而非卡死，自测窗口要够长 |
| ④ | 末端夹爪 | 有现成组件就"采购"（拷 body/joint/equality/actuator + 加左右前缀）；**组件出厂参数 ≠ 调参后参数**——前辈把 equality 调硬了（solref 0.005→0.002 + solimp），抄组件只抄到一半就是"夹不上"事故（见 §8.3） |
| ⑤ | 碰撞体 | 见 §8，整个流程里最容易翻车的一步 |
| ⑥ | 场景 | 桌/箱/方块（free joint）/相机/**home keyframe**（关节+ctrl+物体位姿全套）；`<visual><global offwidth/offheight>` **必须显式设大**（默认 640×480，512 渲染直接报"超出离屏缓冲"） |
| ⑦ | 对照参考工程 | 关节/执行器/约束/sensor 数量逐项 diff，差异要么闭合要么能解释 |

每步的 self_test 至少包含：编译通过、结构数符合预期、2s 物理稳定无发散、
夹爪开合行程达标。

---

## 4. 接 PICO：硬件链路层（照手册 + 三个坑）

 prerequisites：PICO 4 Ultra 头显 + 双手柄，PC 与头显同一局域网。

1. PICO 开开发者模式（设置→通用→关于本机→连点版本号）+ USB 调试
2. PC 装 ADB：`sudo apt install -y adb`，`adb devices` 授权
3. `adb install -r -g XRoboToolkit-PICO-1.1.1.apk`
4. PC 装 Service（deb 包），验证 `/opt/apps/roboticsservice/runService.sh` 存在
5. **启动 Service**：`cd /opt/apps/roboticsservice && nohup bash ./runService.sh &`
   —— 必须**先 cd 再 bash**，直接 `./runService.sh` 会 "Bad substitution"
6. PICO 里开 XRoboToolkit App，选 PC 的 WiFi IP；确认两个开关都 ON：
   **Controller Tracking / Send Tracking Data**（可能被重置回 OFF）
7. xr-robotics 环境跑 SDK 自带示例，确认手柄/头显位姿和按键在刷

**故障速查**：
- 数据全 0（ts=0）→ 头显**摘下=熄屏=App 断连且不自动重连**；重新戴上、
  App 内确认在线，必要时完全退出重开
- 连不上 → PC 侧 `ss -tn | grep 63901` 看端口（63901=头显，60061=本机 SDK）；
  ping 通、端口开、ufw 关还不行，多半是 App 侧陈旧状态，重开重试
  （~150ms WiFi 延迟可能让握手超时，多重试几次）

---

## 5. 遥操作程序：模型接口 + IK 配方 + 按键

### 5.1 模型侧要补的接口（teleop 程序的硬要求）

| 接口 | 为什么 |
|---|---|
| `left/right_teleop_target`（mocap 体，纯视觉） | 每周期把 IK 目标位姿画出来 |
| `*_omnipicker_tcp_link`（与 IK URDF 同名的 body 壳） | `--check-model` 要求 MJCF body 与 URDF frame 同名且位姿一致 |
| 夹爪 TCP site、`{joint}_position` 执行器、`*_gripper_opening` | 控制与状态读取 |

写成一个幂等补丁脚本（H1_build 的 11 号），跑完 `--check-model` 必须过。

### 5.2 IK（placo）配方与坑

- 纯运动学 URDF：删 visual/collision + 追加 TCP link（FK 与 MuJoCo site
  误差应 <0.001mm）
- 任务配方：`posture`(soft 1e-3，钉零空间——**没有它关节会漂到限位**) +
  `trunk`(hard，钉腰头) + 每臂一个 `frame_task`（激活 weight 1 / 冻结 0，
  冻结时换关节空间 hold 任务）
- **坑**：连续 `solve(True)` 后必须 `update_kinematics()`，否则用过期雅可比
  发散；QP 异常/NaN 要自愈（placo 状态拉回 MuJoCo 真值 + 重锚定目标）；
  读 site 前必须 `mj_forward`（reset 后 site_xpos 全 0，锚到世界原点会 NaN）
- mujoco viewer 的 C++ 层**内部占用数字键**（按 2 模型消失、按 3 幽灵渲染，
  帮助表不列）——给 viewer 挑功能键只用字母

### 5.3 启动命令：遥操作 / 数采怎么触发  ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1

```bash
cd H1_build && conda activate xr-robotics

# ① 纯遥操作（不录数据）
python teleop/h1_pico_teleop.py

# ② 数采模式 —— --record 是唯一开关，其余参数都有默认值
python teleop/h1_pico_teleop.py --record \
  --task "Pick up the red cube and put it into the green bin" \
  --dataset-dir data/h1_build_pick \
  --record-fps 30
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--record` | 关 | 数采唯一开关；不加就是纯遥操作 |
| `--task` | 递给记录器的 DEFAULT_TASK（见 §6.2） | 每帧携带的语意文本 |
| `--dataset-dir` | `H1_build/data/h1_build_pick` | 目录已存在→自动续采；不存在→新建 |
| `--repo-id` | `mjq/h1_build_pick` | 数据集标识（info.json/meta 里记录） |
| `--record-fps` | 30 | 录制帧率（按仿真时刻调度，与控制环解耦） |
| `--check-model` | — | 无头自检 MJCF/URDF 接口一致性，改完模型先跑它 |
| `--debug-xr` / `--no-wrist-viewers` | — | 调 XR 数据流 / 关腕部相机窗口 |

**录制过程不需要任何命令**：X 键开关回合、Ctrl-C 退出自动保存并
finalize、重跑同命令自动续采。

### 5.4 按键表（PICO 手柄）

| 按键 | 功能 |
|---|---|
| 按住左/右 **Grip** | 接管对应臂（**相对运动**，手柄↔TCP 对齐，不跳世界坐标） |
| 松开 Grip | 该臂冻结；再按住=重建相对零点 |
| 左/右 **Trigger**（模拟量） | 夹爪开合：不按=全开，扣到底=全闭 |
| 右手 **X** | 录制开关：一按开始 episode，再按保存 |
| 右手 **B** | 双臂松开 + 平滑回 home；录制中=先保存再回零 |
| 右手 **A** | 重置方块；录制中=先**丢弃**本回合再重置 |
| 头显朝向 | 自动映射机器人头部相机 |

---

## 6. 数采：LeRobotDataset v3.0 记录器

### 6.1 每帧采集的四部分数据（features 设计）

| 数据 | 数据集字段 | 内容 |
|---|---|---|
| 图像 | `observation.images.head_rgb / left_wrist_rgb / right_wrist_rgb` | 三相机 512×512（dtype=video，存 mp4） |
| 关节角度 | `observation.state` | float32 **[20]** = 上身 18 关节 qpos + 左右夹爪开合测量值 |
| 控制指令 | `action` | float32 **[20]** = 18 路关节位置 ctrl + 2 路夹爪 ctrl |
| 语意 | `task` 字符串 | 任务描述，存 `meta/tasks.parquet`，每帧经 `task_index` 关联 |

fps=30（按仿真时刻调度，与 100Hz 控制环解耦）；四部分在**同一个
MuJoCo 仿真时刻**采样，天然时间对齐。

**关键细节——"控制指令"录的是什么**：不是 PICO 手柄的原始信号
（手柄位姿、Grip/Trigger 模拟量**不落盘**），而是它们经 IK/映射后
**最终写给 MuJoCo 执行器的 ctrl 目标**。这是刻意设计：

- 训练时：policy 输入 = state + 图像 + 语意，监督标签 = action；
- 推理时：policy 输出同维度 action 直接写给机器人——与遥操作写入的
  是**同一种量**，闭环才成立；
- 原始手柄数据是"人怎么操作"的过程量，策略不需要复刻，不采。

另有 `timestamp / frame_index / episode_index / index / task_index` 五列
由 lerobot 自动生成的索引元数据，不算采集内容。

### 6.2 语意（task）在哪改、怎么生效

两处，推荐①（不改代码）：

```bash
# ① 启动命令行指定（每次会话生效）
python teleop/h1_pico_teleop.py --record --task "Pick up the blue cube"
```

```python
# ② 改默认值：teleop/h1_dataset_recorder.py:49
DEFAULT_TASK = "Pick up the red cube and put it into the green bin"
```

teleop 脚本 import 这个常量做 `--task` 的 default，改这一处两边默认值都变。

**三个注意**：
1. task 是**每次会话一个值**，构造记录器时定死，该次运行所有 episode
   共用——中途不能换，要换就退出重开。
2. **续采换 task = 多任务数据集**：重跑同命令自动 resume 已有目录，
   lerobot 0.5.2 遇到新 task 字符串会追加为新 `task_index`（老回合挂
   task 0、新回合挂 task 1）——v3.0 合法，但想让数据集保持单一任务
   纯净就换个新目录：`--dataset-dir data/h1_build_pick_blue`。
3. 任务文本用**英文**更稳——smolvla 的语言条件化对英文语料对齐最好。

### 6.3 记录器工程要点

- **与 XR 解耦**：记录器只依赖 mujoco/numpy/lerobot，无头可独立验证
  （先跑 `h1_record_check.py` 全绿，实机零改动）
- 图像临时 PNG 落盘不占内存，`save_episode` 在**后台线程**串行编码，
  不卡控制环；退出前必须 `finalize()`
- **空壳自愈**：`create()` 后一集未存就退出会留"只有 info.json 的空壳"，
  下次 `resume()` 读不到 tasks.parquet 会转去 HF Hub 报 401——用
  "tasks.parquet 存在与否"判断可否续采，空壳直接删重建
- 丢弃 episode 前先 `image_writer.wait_until_done()`，否则异步残留写
  污染下一回合同名 PNG

### 6.4 v3.0 格式校验清单（磁盘布局 + 语义约定）

```
meta/info.json          codebase_version=v3.0; total_episodes/frames/tasks;
                        data_path/video_path/chunks_size/features/splits
                        （无 total_videos，是 total_tasks）
meta/tasks.parquet      两列 [task_index, task]（task 是普通列，不是行索引）
meta/episodes/chunk-XXX/file-XXX.parquet
                        每集 length/tasks/stats + data 与 video 的 chunk/file 指针
data/chunk-XXX/file-XXX.parquet
                        列 = 各特征 + (timestamp, frame_index, episode_index,
                        index, task_index)
videos/<video_key>/chunk-XXX/file-XXX.mp4   每相机每 chunk 一个 mp4
```

**语义约定（写校验脚本最容易搞混的三件事）**：
- `dataset_to_index` 是**开区间**（to = from + length）
- `index` 是**全局连续**行号；`frame_index` 才每集从 0 重数
- `timestamp` 是**每集内**耗时（每集从 0 起），帧距 = 1/fps

回读校验（lerobot 0.5.2）：len==total_frames、首帧可解码（**float32 [0,1]
的 torch 张量**，不是 uint8）、task 文本往返一致、末帧属于最后一集。
采集回合数不需要预设——X 键开关式，Ctrl-C 退出自动 finalize，重跑同命令
自动续采。

---

## 7. 质量关卡：每层都有可重跑的自检

| 关卡 | 验证什么 |
|---|---|
| 每步脚本 self_test | 编译、结构数、物理稳定、夹爪行程 |
| `--check-model` | MJCF/URDF 名称与 TCP 位姿一致（模型能被 teleop 消费） |
| 无头 record check | 记录链路端到端（脚本驱动+回读校验），实机前必过 |
| 接触几何指纹 | 手指碰撞体 vs 参考工程逐点距离（开度×位置网格） |
| 动力学参数 diff | opt/接触/equality/执行器/关节/质量逐项打印对比 |
| v3.0 格式校验 | §6.4 清单 |

---

## 8. 碰撞与手感调参（最容易翻车的部分，单独立节）

### 8.1 结构：视觉/碰撞分离 + 自碰免疫

```
visual:           contype=0 conaffinity=0   纯外观，和谁都不碰
collision(本体):  contype=0 conaffinity=1   只碰环境(contype=1)，机器人之间永不自碰
finger_collision: 同上 + 高摩擦高硬度       抓取专用
```

**自碰撞靠位掩码解决，不靠碰撞体形状**——这就是手指能放心用 STL 凸包
（`type="mesh"`）的原因。四连杆耦合件天然交叉，若用实体碰撞必然自卡；
位掩码方案下随便用。

### 8.2 形状：手指必须 mesh，本体可以 box

- 本体/躯干：48% AABB 盒（便宜、够用）
- **手指+耦合杆：STL 凸包**——AABB 盒比真手指平均胖 6mm，肉眼看没碰到、
  盒角已在硬推物体：表现就是"穿模"和"弹飞"（实测差异 6.35mm，78% 采样点偏胖）

### 8.3 参数：五件套 + 求解器三件

```
finger_collision: margin=0.001  priority=2  condim=6
                  friction="2.0 0.05 0.005"
                  solref="0.001 1"  solimp="0.99 0.9999 0.0001"
<option>:         cone="elliptic"  noslip_iterations="10"
夹爪 equality:    solref="0.002 1"  solimp="0.99 0.999 0.0001"   ← 要硬!
```

- equality 软（solimp 留默认 0.9 0.95 0.001）→ 手指被物体接触力顶回，
  "夹不拢"；noslip=0 → 物体在指间蠕滑，抓取发"酥"；pyramidal 锥 →
  挤压摩擦失真
- **采购组件的出厂参数通常和参考工程最终调好的参数不一样**——对照要
  对照"最终模型"，逐参数 diff，别信组件默认值

### 8.4 测试方法论

- ❌ **别用"瞬移物体进指间再闭合"测抓取质量**——那测的是求解器暴力分离
  初始重叠的烈度（同参数下连参考工程都能弹飞 6m/s，且是混沌不可复现）
- ✅ 几何问题用**几何指纹**（网格化 mj_geomDistance 逐点对比参考工程）
- ✅ 参数问题用**参数 diff**（逐项打印，差异行自己跳出来）
- 夹爪"夹住物体后有阻力、停在半开"是**正常物理**；不正常的是回弹、蠕滑、穿模

---

## 9. 常见故障速查表

| 现象 | 根因 | 修法 |
|---|---|---|
| URDF 转 MJCF 报 mesh 找不到 | `package://` 路径 | 改相对路径 |
| 关节全不动/姿态怪 | SolidWorks 限位全 0 | 逐关节补限位 |
| 夹爪"蠕变慢"误判卡死 | forcerange 限幅 | 加大 ±1.2；自测窗口 > 收敛时间 |
| `cv2.namedWindow` not implemented | opencv 双版本冲突 | 卸干净只装 GUI 版 |
| recorder 初始化报"超出离屏缓冲" | 默认 640×480 | `<visual><global offwidth/offheight>` 设 3840×2160 |
| `--check-model` TCP frame 不一致 | MJCF 缺同名 body | 从 TCP site 克隆空 body（幂等补丁） |
| resume 报 401/读不到 tasks.parquet | 空壳数据集 | 判 tasks.parquet 存在与否，空壳删重建 |
| 抓取穿模/弹飞 | 手指碰撞体是 AABB 盒 | 改 `type="mesh"` + §8.3 参数 |
| 夹不上、有阻力 | equality 软 + noslip=0 + pyramidal 锥 | §8.3 三件套 |
| XR 数据全 0 | 摘头显断连 | 重戴+检查两个开关；`ss -tn \| grep 63901` |
| Service 起不来 | 相对路径/解释器 | `cd /opt/apps/roboticsservice && bash ./runService.sh` |
| viewer 里模型消失无异常 | 数字键被 C++ 层占用 | 功能键只用字母 |
| IK 静默不动 | 目标 NaN / 局部极小 | 180° 旋转求四元数用 Shepperd 公式；目标连续插值不要瞬移 |

---

## 10. 收尾：数据 → 训练

数据集过了 §6.4 校验即可进训练。完整训练程序：
`H1_build/train/h1_smolvla_training.py`（参照 SO101 双臂版，纯 Python
构造配置，`SMOKE=True` 一键切 2000 步冒烟）。

**一个必须知道的结构性差异——rename_map**：smolvla_base 预训练的视觉
特征键是 `camera1/camera2/camera3`，本工程数据集键是语义命名
（`left_wrist_rgb/right_wrist_rgb/head_rgb`），微调时必须映射对齐，
否则预训练权重接不上：

```python
rename_map = {
    "observation.images.left_wrist_rgb":  "observation.images.camera1",   # 左腕
    "observation.images.right_wrist_rgb": "observation.images.camera2",   # 右腕
    "observation.images.head_rgb":        "observation.images.camera3",   # 头
}
```

等价 CLI 冒烟命令（正式训练 steps=20000 ≈ 7.2 遍）：

```bash
conda activate lerobot312
lerobot-train --policy.type=smolvla --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=mjq/h1_build_pick \
  --dataset.root=<数据集路径> \
  --rename_map observation.images.left_wrist_rgb=observation.images.camera1 \
  --rename_map observation.images.right_wrist_rgb=observation.images.camera2 \
  --rename_map observation.images.head_rgb=observation.images.camera3 \
  --output-dir=results/smolvla_test --steps=2000     # 先冒烟
```

小批量先跑通链路，再扩采到百回合规模正式训练。
