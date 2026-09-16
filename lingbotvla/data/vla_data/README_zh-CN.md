# 自定义下游任务数据集构建

> 本文档是 [README.md](README.md) 的中文翻译,代码、字段名、配置与命令均保持原样,仅翻译说明文字。

本指南介绍如何为后训练(post-training)准备下游 VLA 数据。默认数据路径为**单个 LeRobot 数据集**。如果要在多个数据集上训练,可设置 `data.data_name: multi`,并通过 `data.train_path` 传入一个文本列表。

## 1. 准备 LeRobot 数据集

LingBot-VLA 2.0 通过 [LeRobot](https://github.com/huggingface/lerobot) 库(`LeRobotDataset`)加载下游数据。每个数据集条目可以是 HuggingFace repo id,也可以是本地 LeRobot 数据集目录。

**LeRobot v2.1** 与 **LeRobot v3.0** 两种目录结构均直接支持,无需在训练前合并数据集,也无需将 v2.1 数据转换为 v3.0。

## 2. 准备数据集输入

### 单个数据集(默认)

对于单个 LeRobot 数据集,将 `data.data_name` 设为 robot config 名称,并将 `data.train_path` 设为数据集的 repo id 或本地目录。

以 RoboTwin 为例,使用 `robotwin`,加载器会解析为:

```text
configs/robot_configs/robotwin.yaml
```

对应的 VLA 配置应使用:

```yaml
data:
  datasets_type: vla           # 数据集构建器类型;VLA 后训练固定用 "vla"
  data_name: robotwin          # 单个数据集:填 robot config 名称(去掉 .yaml),加载器据此找 configs/robot_configs/robotwin.yaml
  train_path: /path/to/lerobot_dataset           # LeRobot 数据集的 repo id 或本地目录路径
  robot_config_root: ./configs/robot_configs     # robot config YAML 所在的根目录
  joints:                      # 声明统一特征空间里各关节类型的"槽位维度"(槽位须 >= 实际数据维度,不足补零 padding)
    - arm.position: 14         # 臂关节槽位:14(预训练即14维,用于对齐);RoboTwin 实际左臂6+右臂6=12,末尾补 2 零
    - end.position: 14         # 末端位姿槽位:14(RoboTwin 不用 end,全 padding,仅为对齐预训练 head 维度)
    - effector.position: 2     # 夹爪槽位:2;RoboTwin 实际左夹爪1+右夹爪1=2,正好填满
  cameras:                     # 相机名列表(须与 robot config 的 images 映射、模型输入一致)
    - camera_top               # 顶部/第三视角相机
    - camera_wrist_left        # 左腕相机
    - camera_wrist_right       # 右腕相机
  prompt_type: global          # 任务指令(prompt)类型:"global"=整段任务用一条全局指令;"subtask"/"both" 见文档
```

### 多个数据集

对于多个 LeRobot 数据集,设置 `data.data_name: multi`,并将所有数据集写入一个文本文件,`data.train_path` 指向该文件。

每个非空行包含两列,以空格分隔:

```text
<robot_config_name> <lerobot_repo_or_local_path>
```

示例 `assets/training_data/robotwin.txt`:

```text
robotwin /path/to/lerobot_task_a
robotwin /path/to/lerobot_task_b
robotwin /path/to/lerobot_task_c
```

对应的 VLA 配置应使用:

```yaml
data:
  datasets_type: vla           # 数据集构建器类型;VLA 后训练固定用 "vla"
  data_name: multi             # 多数据集模式:固定填 "multi"(每个数据集的 robot config 由 train_path 指向的列表文件指定)
  train_path: assets/training_data/robotwin.txt   # 指向数据集列表文件,每行格式:<robot_config_name> <数据集路径>
  robot_config_root: ./configs/robot_configs       # robot config YAML 所在的根目录
  joints:                      # 统一特征空间各关节类型的槽位维度(多数据集时所有数据集共用同一套槽位定义)
    - arm.position: 14         # 臂关节槽位(对齐预训练14维);实际 6+6=12,末尾补 2 零
    - end.position: 14         # 末端位姿槽位(RoboTwin 不用,全 padding 占位对齐)
    - effector.position: 2     # 夹爪槽位;实际 1+1=2 正好填满
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  prompt_type: global
```

`MultiVLADataset` 会为每一行实例化一个 `VLADataset`,并在运行时将它们拼接起来。这取代了过去"先合并数据集"的工作流。

## 3. 准备 Robot Config

robot config 用于将原始 LeRobot 特征名映射到 LingBot-VLA 2.0 所使用的统一特征空间。对于当前的 RoboTwin 配置,使用:

```text
configs/robot_configs/robotwin.yaml
```

去掉 `.yaml` 后的文件名,在单个数据集时作为 `data.data_name`,在多数据集列表中作为第一列:

```text
robotwin /path/to/lerobot_dataset
```

### States(状态)

`states` 将原始 observation 键映射到统一的状态特征。来自同一原始张量的多个切片会按顺序拼接。

当前 RoboTwin 的状态映射:

```yaml
states:                                     # 把原始 observation 键映射到统一的状态特征(多个切片按顺序拼接)
  - observation.state.arm.position:         # 统一状态里的"臂关节位置"特征(需在 data.joints 中声明 arm.position)
      origin_keys:                          # 来自哪些原始张量及其切片(可多个,按顺序拼接)
        - observation.state:                # 左臂关节 [0:6),共 6 维
            start: 0                        # 切片起点(闭区间,包含该索引)
            end: 6                          # 切片终点(开区间,不包含该索引,即取到索引 5)
        - observation.state:                # 右臂关节 [7:13),共 6 维(跳过索引 6 = 左夹爪)
            start: 7
            end: 13

  - observation.state.effector.position:    # 统一状态里的"夹爪位置"特征(需在 data.joints 中声明 effector.position)
      origin_keys:
        - observation.state:                # 左夹爪 [6:7),1 维
            start: 6
            end: 7
        - observation.state:                # 右夹爪 [13:14),1 维
            start: 13
            end: 14
```

含义如下:

| 统一状态特征 | 原始切片 | 总维度 |
|---|---|---|
| `observation.state.arm.position` | `[0:6)` + `[7:13)` | 12 |
| `observation.state.effector.position` | `[6:7)` + `[13:14)` | 2 |

### Actions(动作)

`actions` 的结构与 `states` 相同,但映射的是原始动作张量到统一动作特征。`subtract_state: False` 表示模型学习绝对动作,而非相对于状态的变化量(delta)。

当前 RoboTwin 的动作映射:

```yaml
actions:                                    # 把原始 action 键映射到统一的动作特征(结构与 states 相同)
  - action.arm.position:                    # 统一动作里的"臂关节位置"目标(需在 data.joints 中声明 arm.position)
      origin_keys:
        - action:                           # 左臂动作 [0:6)
            start: 0
            end: 6
        - action:                           # 右臂动作 [7:13)
            start: 7
            end: 13
      subtract_state: False                 # 是否学相对动作:True=目标减去当前状态(学增量 delta);False=学绝对动作。RoboTwin 用 False;真机建议 arm 用 True

  - action.effector.position:               # 统一动作里的"夹爪位置"目标
      origin_keys:
        - action:                           # 左夹爪动作 [6:7)
            start: 6
            end: 7
        - action:                           # 右夹爪动作 [13:14)
            start: 13
            end: 14
      subtract_state: False                 # 夹爪:RoboTwin 与真机都建议 False(学绝对开合量)
```

> **注意**:在真机数据上训练时,建议将 `action.arm.position` 的 `subtract_state` 设为 `True`,将 `action.effector.position` 的设为 `False`。完整示例见 [`configs/robot_configs/agilex_cobot_magic.yaml`](../../../configs/robot_configs/agilex_cobot_magic.yaml)。

### Images(图像)

`images` 将原始相机键映射到 VLA 训练配置中声明的统一相机名。

当前 RoboTwin 的相机映射:

```yaml
images:                                            # 把原始相机键重命名为统一相机名(统一名须在 data.cameras 中声明)
  - observation.images.camera_top:                 # 统一相机名:顶部/第三视角相机
      origin_keys: observation.images.cam_high      # 原始数据里的相机键名(RoboTwin 原名为 cam_high)
  - observation.images.camera_wrist_left:          # 统一相机名:左腕相机
      origin_keys: observation.images.cam_left_wrist
  - observation.images.camera_wrist_right:         # 统一相机名:右腕相机
      origin_keys: observation.images.cam_right_wrist
```

如果原始键已经与目标键相同,可以使用简写形式:

```yaml
images:
  - observation.images.camera_top   # 原始键已与统一相机名相同,可省略 origin_keys,直接简写
```

### Normalization Stats(归一化统计)

robot config 还会指向归一化文件:

```yaml
norm_stats: assets/norm_stats/robotwin.json   # 归一化统计文件路径(由 compute_norm_stats.py 生成);训练时从 robot config 读取此路径
```

训练时会从 robot config 读取该路径。重新计算 norm stats 后,请确保该字段指向已生成的 JSON 文件。

### 与 VLA 训练配置保持一致

robot config 中使用的关节类型(joint type)和相机名必须已在 VLA 训练配置中声明。

示例:

```yaml
# configs/vla/robotwin/robotwin.yaml
data:
  datasets_type: vla           # 数据集构建器类型;VLA 后训练固定用 "vla"
  data_name: robotwin          # 单个数据集:使用 robot config 名称;仅在使用数据集列表时才用 "multi"。
  train_path: /path/to/lerobot_dataset           # LeRobot 数据集目录或 repo id
  robot_config_root: ./configs/robot_configs     # robot config YAML 所在根目录
  joints:                      # 统一特征空间各关节类型的槽位维度(槽位 >= 实际数据维度,不足补零 padding)
    - arm.position: 14         # 臂关节槽位(对齐预训练14维);实际 6+6=12,末尾补 2 零
    - end.position: 14         # 末端位姿槽位(RoboTwin 不用 end,全 padding 占位对齐)
    - effector.position: 2     # 夹爪槽位;实际 1+1=2 正好填满
  cameras:                     # 相机名(须与 robot config 的 images 映射一致)
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  prompt_type: global          # 任务指令类型:global / subtask / both
```

规则:

- robot config 中的 `observation.state.<joint_type>` 和 `action.<joint_type>` 必须在 `data.joints` 中有对应条目。
- robot config 中的 `observation.images.<camera_name>` 必须列在 `data.cameras` 中。
- 配置的关节维度应大于或等于拼接后的原始切片维度。对于 RoboTwin,`arm.position` 使用 12 维,`effector.position` 使用 2 维。
- 训练配置中可以存在额外的关节条目,但只有在 robot config 将数据映射到它们时才会被使用。在 RoboTwin 示例中,保留 `end.position: 14` 是为了与预训练配置及模型的动作/状态头维度对齐。

> 完整示例见 `configs/robot_configs/robotwin.yaml`。

> **重要:**
> - states(`observation.state.<joint_type>`)和 actions(`action.<joint_type>`)中使用的 `<joint_type>` 必须在 VLA 训练配置的 `data.joints` 中定义。
> - images 部分(`observation.images.<camera_name>`)中的相机名必须列在 `data.cameras` 中。

例如,如果 `configs/vla/robotwin/robotwin.yaml` 声明了 `joints: [{arm.position: 14}, {end.position: 14}, {effector.position: 2}]` 和 `cameras: [camera_top, camera_wrist_left, camera_wrist_right]`,那么 robot config 中只有这些关节类型和相机名是有效的。使用未定义的关节类型或相机名会在运行时抛出 `ValueError`。

> **注意:** 可以通过在 `data.joints` 中添加新条目来定义 `arm.position` 和 `effector.position` 之外的其他关节类型。如果新增末端执行器(EEF)维度,建议学习**绝对动作**(`subtract_state: False`),因为目前不支持相对旋转的计算。

## 4. 计算归一化统计量

直接使用 robot 训练配置即可,无需合并或格式转换步骤。

对于单个 LeRobot 数据集目录,将 `data.data_name` 设为 robot config 名称,并将数据集目录作为 `data.train_path` 传入:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name robotwin \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/robotwin.json \
  --data.data_ratio_for_norm_compute 1
```

示例:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name robotwin \
  --data.train_path /path/to/beat_block_hammer-aloha-agilex_randomized_500-1000/ \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path debug.json \
  --data.data_ratio_for_norm_compute 1
```

对于多数据集列表,保持 `data.data_name: multi`,并将列表文件作为 `data.train_path` 传入:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/robotwin.txt \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/robotwin.json \
  --data.data_ratio_for_norm_compute 1
```

可选:只为列表中指定的 robot config 名称计算统计量:

```bash
--data.robot_name robotwin
```

输出的 JSON 应与 `configs/robot_configs/robotwin.yaml` 中 `norm_stats` 的路径一致。

## 5. 训练

当 LeRobot 数据集、robot config 和 norm stats 都准备好后,即可开始后训练。默认用法是单个数据集,`data.data_name` 设为 robot config 名称。

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.norm_stats_file assets/norm_stats/robotwin.json
```

对于单个数据集,可通过命令行覆盖路径:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name robotwin \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_stats_file assets/norm_stats/robotwin.json \
  --train.output_dir output/
```

对于多个数据集,设置 `--data.data_name multi`,并通过 `--data.train_path` 传入文本列表:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/robotwin.txt \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_stats_file assets/norm_stats/robotwin.json \
  --train.output_dir output/
```

## 快速检查清单

- 数据集条目是 LeRobot v2.1 或 v3.0 的仓库/本地目录。
- 单个数据集:`data.data_name` 是 robot config 名称(例如 `robotwin`),`data.train_path` 指向一个 LeRobot 数据集目录。
- 多个数据集:`data.data_name` 为 `multi`,`data.train_path` 指向一个文本文件,其第一列为 robot config 名称(例如 `robotwin`)。
- `configs/robot_configs/robotwin.yaml` 存在。
- robot config 中的 `norm_stats` 指向一个已存在的 JSON 文件。
- 训练时,如果想覆盖或显式指定 norm stats JSON,可传入 `--data.norm_stats_file path/to/norm_stats.json`。
- `data.joints` 和 `data.cameras` 包含了 robot config 用到的所有统一关节和相机。
