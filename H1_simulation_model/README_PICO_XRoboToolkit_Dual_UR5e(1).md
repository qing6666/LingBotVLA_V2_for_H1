# PICO 4 Ultra + XRoboToolkit 双 UR5e 遥操作复现指南

本文档记录从 **PICO 4 Ultra 开发环境配置**、**XRoboToolkit 安装**、**PICO 数据接收验证**，到运行官方 **双 UR5e MuJoCo 遥操作 Demo** 的完整流程。MuJoCo学习可以通过https://space.bilibili.com/399384226?spm_id_from=333.337.search-card.all.click 复现几个demo即可

---

## 1. 系统架构

```text
PICO 4 Ultra
    │
    │ Wi-Fi：头显、手柄位姿及按键数据
    ▼
XRoboToolkit PC Service
    │
    │ xrobotoolkit_sdk Python Binding
    ▼
XRoboToolkit-Teleop-Sample-Python
    │
    │ 位姿映射 + 逆运动学
    ▼
MuJoCo 双 UR5e 仿真
```

---

## 2. 环境要求

### Ubuntu 电脑

- Ubuntu 22.04 x86_64
- Conda
- Git
- ADB
- PICO 与电脑连接到同一个局域网

### PICO 设备

- PICO 4 Ultra
- 已开启开发者模式
- 已开启 USB 调试
- 已安装 XRoboToolkit APK

---

## 3. 准备安装包

需要准备以下两个文件：
下载地址在：https://github.com/XR-Robotics
```text
XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb
XRoboToolkit-PICO-1.1.1.apk
```

实际下载文件名可能使用下划线，例如：

```text
XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

以下命令均以实际文件名为准。

假设文件位于：

```text
~/Downloads/
```

中文系统中也可能位于：

```text
~/下载/
```

检查文件：

```bash
ls -lh ~/Downloads/*XRoboToolkit*
```

若文件位于中文下载目录：

```bash
ls -lh ~/下载/*XRoboToolkit*
```

---

## 4. 开启 PICO 4 Ultra 开发者模式

在 PICO 中进入：

```text
设置
→ 通用
→ 关于本机
→ 连续点击“软件版本号”
→ 打开开发者选项
→ 开启 USB 调试
```

还可以在：

```text
设置 → 通用
```

将“电脑互联自动发现”关闭后重新开启，以刷新设备发现状态。

---

## 5. Ubuntu 安装 ADB 和基础工具

```bash
sudo apt update

sudo apt install -y \
    adb \
    git \
    build-essential \
    cmake \
    pkg-config \
    curl
```

检查 ADB：

```bash
adb version
```

使用 USB 数据线连接 PICO，然后执行：

```bash
adb devices
```

第一次连接时，PICO 内会弹出 USB 调试授权窗口。

勾选：

```text
始终允许此计算机
```

正常输出类似：

```text
List of devices attached
XXXXXXXXXXXX    device
```

如果显示：

```text
unauthorized
```

执行：

```bash
adb kill-server
adb start-server
adb devices
```

然后重新在 PICO 中确认授权。

---

## 6. 安装 XRoboToolkit APK

进入 APK 所在目录：

```bash
cd ~/Downloads
```

中文系统可使用：

```bash
cd ~/下载
```

安装：

```bash
adb install -r -g XRoboToolkit-PICO-1.1.1.apk
```

参数含义：

```text
-r：覆盖安装已有版本
-g：安装时授予运行权限
```

成功时显示：

```text
Success
```

检查安装结果：

```bash
adb shell pm list packages | grep -i robot
```

安装完成后可以拔掉 USB。后续数据通过 Wi-Fi 发送。

---

## 7. 安装 XRoboToolkit PC Service

进入 DEB 文件所在目录：

```bash
cd ~/Downloads
```

或者：

```bash
cd ~/下载
```

先检查实际文件名：

```bash
ls *PC*Service*.deb
```

安装：

```bash
sudo apt install ./XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

若文件名为连字符版本，则执行：

```bash
sudo apt install ./XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb
```

安装日志中出现下面内容，说明安装成功：

```text
正在设置 roboticsservice (1.0.0.0) ...
```

如果最后出现 `_apt` 无权限访问本地文件的提示，一般只是 apt 读取用户目录文件时的权限提示，不代表安装失败。

检查软件包状态：

```bash
dpkg -l | grep roboticsservice
```

或者：

```bash
dpkg -s roboticsservice
```

检查安装目录：

```bash
ls -lh /opt/apps/roboticsservice/
```

应包含：

```text
runService.sh
```

---

## 8. 启动 PC Service （后续只要用到，都得先执行这一步骤）
![alt text](image.png)
执行：

```bash
/opt/apps/roboticsservice/runService.sh
```

启动时可能输出：

```text
/opt/ros/humble/opt/rviz_ogre_vendor/lib:...
/opt/apps/roboticsservice/plugins/:
/opt/apps/roboticsservice/qml/:
release mode
```

脚本执行后终端重新出现命令提示符并不一定表示服务停止。该脚本可能将服务放到后台运行。

确认进程：

```bash
pgrep -af "/opt/apps/roboticsservice"
```

也可以执行：

```bash
ps aux | grep -Ei "roboticsservice|robotics|pxrea" | grep -v grep
```
![alt text](image-1.png)
如果提示没有执行权限：

```bash
sudo chmod +x /opt/apps/roboticsservice/runService.sh
/opt/apps/roboticsservice/runService.sh
```

> PC Service 同一时间只运行一个实例。不要同时从桌面图标和终端重复启动。

---

## 9. 确认电脑局域网 IP

```bash
hostname -I
```

例如：

```text
192.168.1.105
```

应使用与 PICO 位于同一局域网的地址，不要使用：

```text
127.0.0.1
172.17.x.x
Docker 网卡地址
虚拟机网卡地址
```

---

## 10. PICO 连接 PC Service（具体的图片可以在pico里面相册最新进行观看）

保持 PC Service 正在运行，然后在 PICO 中打开：

```text
XRoboToolkit
```

正常情况下，PICO 会自动发现 Ubuntu 电脑。

使用手柄射线选中电脑 IP，并按 Trigger 确认。

如果没有自动发现，可以手动进入：

```text
Network
→ PC Service
→ Enter
```

输入 Ubuntu 局域网 IP。

连接成功时，PICO 端可能显示：

```text
PC connection established
version packet sent successfully
```

主界面状态应进入：

```text
WORKING
```

PICO 端确认以下选项：

```text
Controller Tracking：ON
Send Tracking Data：ON
Hand Tracking：按需要开启
Motion Tracker：没有设备时设为 None
```

---

## 11. 下载官方 Python 遥操作仓库

本文统一使用以下目录：

```text
~/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python
```

执行：

```bash
mkdir -p ~/XRoboToolkit
cd ~/XRoboToolkit

git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
cd XRoboToolkit-Teleop-Sample-Python
```

记录当前版本：

```bash
git rev-parse HEAD | tee REPRODUCED_COMMIT.txt
git status
```

仓库主要包含：

```text
assets/
scripts/
xrobotoolkit_teleop/
setup_conda.sh
teleop_details.md
```

---

## 12. 创建 Conda 环境并安装依赖

使用官方脚本创建环境：

```bash
cd ~/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python

bash setup_conda.sh --conda xr-robotics
conda activate xr-robotics
bash setup_conda.sh --install
```

注意：

```text
bash setup_conda.sh --conda xr-robotics
```

可能会删除并重新创建同名环境。

如果环境已经创建完成，只执行：

```bash
conda activate xr-robotics
bash setup_conda.sh --install
```

安装脚本会创建：

```text
dependencies/XRoboToolkit-PC-Service-Pybind
```

并编译安装：

```text
xrobotoolkit_sdk
```

---

## 13. 检查 Python Binding

```bash
conda activate xr-robotics

python -c "import xrobotoolkit_sdk as xrt; print('xrobotoolkit_sdk 导入成功')"
```

正常输出：

```text
xrobotoolkit_sdk 导入成功
```

查看模块位置：

```bash
python -c "import xrobotoolkit_sdk as xrt; print(xrt.__file__)"
```

同时检查仿真依赖：

```bash
python -c "import mujoco; print('mujoco:', mujoco.__version__)"
python -c "import placo; print('placo 导入成功')"
python -c "import xrobotoolkit_teleop; print('teleop package 导入成功')"
```

---

## 14. 运行官方 PICO 数据连续打印程序

保持以下状态：

```text
PC Service：已运行
PICO：PC connection established / WORKING
Controller Tracking：ON
Send Tracking Data：ON
```

执行：

```bash
cd ~/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python
conda activate xr-robotics

python dependencies/XRoboToolkit-PC-Service-Pybind/examples/run_binding_continuous.py
```

程序会持续打印：

- 左右手柄位姿
- 头显位姿
- Trigger
- Grip
- 摇杆
- A/B/X/Y 按键
- 菜单键
- 手部追踪关节状态
- 时间戳
- Motion Tracker 数量

位姿格式为：

```text
[x, y, z, qx, qy, qz, qw]
```

其中：

```text
x, y, z          位置
qx, qy, qz, qw   旋转四元数
```

正常输出示例：

```text
--- Iteration 74 ---
Left Controller Pose: [-0.4448, -0.2462, -0.1587, -0.1356, 0.1861, -0.4019, 0.8863]
Right Controller Pose: [-0.3997, -0.2466, -0.2188, -0.0479, 0.0125, 0.4128, 0.9095]
Headset Pose: [-0.1592, 0.0618, 0.0773, -0.2871, 0.0969, 0.0435, 0.9520]
Left Trigger: 0.0
Right Trigger: 0.0
Left Grip: 0.0
Right Grip: 0.0
Left Axis (X, Y): [0.0, 0.0]
Right Axis (X, Y): [0.0, 0.0]
Timestamp (ns): 1785294030785516032
Number of Motion Trackers: 0
```

验证项目：

```text
移动左手柄       → Left Controller Pose 变化
移动右手柄       → Right Controller Pose 变化
转动或移动头显   → Headset Pose 变化
按左/右 Grip     → 对应 Grip 从 0 向 1 变化
按左/右 Trigger  → 对应 Trigger 从 0 向 1 变化
推动摇杆         → Axis 数值变化
按 A/B/X/Y       → 对应布尔值变为 True
```

如果未开启裸手追踪，Hand State 可能是重复的默认值，可以忽略。

没有连接 PICO Motion Tracker 时：

```text
Number of Motion Trackers: 0
```

属于正常情况。

---

## 15. 查看仓库中的官方仿真 Demo

```bash
cd ~/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python

find scripts/simulation \
    -maxdepth 1 \
    -type f \
    -name "*.py" \
    -printf "%f\n" \
    | sort
```

常见示例包括：

```bash
# 双 UR5e MuJoCo
python scripts/simulation/teleop_dual_ur5e_mujoco.py

# X7S Placo 运动学可视化
python scripts/simulation/teleop_x7s_placo.py

# Shadow Hand MuJoCo 灵巧手
python scripts/simulation/teleop_shadow_hand_mujoco.py

# Inspire Hand Placo 可视化
python scripts/simulation/teleop_inspire_hand_placo.py
```

---

## 16. 运行双 UR5e MuJoCo 遥操作 Demo

### 16.1 启动前检查

确认 PC Service 正在运行：

```bash
pgrep -af "/opt/apps/roboticsservice"
```

确认 PICO 状态：

```text
PC connection established
WORKING
Controller Tracking：ON
Send Tracking Data：ON
```

进入项目环境：

```bash
conda activate xr-robotics
cd ~/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python
```

检查脚本：

```bash
ls -lh scripts/simulation/teleop_dual_ur5e_mujoco.py
```

### 16.2 启动 Demo

```bash
python scripts/simulation/teleop_dual_ur5e_mujoco.py
```

如需保存日志：

```bash
python scripts/simulation/teleop_dual_ur5e_mujoco.py \
    2>&1 | tee dual_ur5e_run.log
```

正常情况下会打开 MuJoCo 窗口，并显示双 UR5e 模型。

---

## 17. 双 UR5e 操作方法

### 左机械臂

```text
把左手柄放到舒适位置
→ 按住左 Grip
→ 缓慢移动或旋转左手柄
→ 左机械臂跟随
→ 松开左 Grip
→ 左机械臂停止跟随
```

### 右机械臂

```text
把右手柄放到舒适位置
→ 按住右 Grip
→ 缓慢移动或旋转右手柄
→ 右机械臂跟随
→ 松开右 Grip
→ 右机械臂停止跟随
```

Grip 相当于“死人开关”：

```text
按住 Grip   → 激活对应机械臂
松开 Grip   → 停止对应机械臂跟随
```

每次重新按下 Grip 时，程序会以当前手柄位姿和机器人末端位姿重新建立相对参考零点。

### 推荐验收顺序

```text
1. 不按 Grip，确认两台机械臂保持静止
2. 只按右 Grip，测试右臂平移
3. 测试右手柄旋转
4. 松开右 Grip，确认右臂停止
5. 只按左 Grip，重复测试
6. 最后同时控制双臂
```

> Trigger、B 键和摇杆的具体功能取决于当前运行的 Demo 配置。仓库的部分硬件或带夹爪 Demo 会使用 Trigger 控制夹爪、B 键控制数据记录；双 UR5e MuJoCo 脚本是否启用这些功能，应以当前脚本中的配置为准。

---

## 18. 常见控制映射

XRoboToolkit 仓库中的常见设计如下：

| PICO 输入 | 常见功能 |
|---|---|
| 按住左 Grip | 激活左机械臂 |
| 按住右 Grip | 激活右机械臂 |
| 松开 Grip | 停止对应机械臂跟随 |
| 左 Trigger | 部分 Demo 中控制左夹爪 |
| 右 Trigger | 部分 Demo 中控制右夹爪 |
| B 键 | 部分 Demo 中开始或停止数据记录 |
| 右摇杆按下 | 部分 Demo 中放弃当前记录 |

判断当前 Demo 是否启用某个按键，应查看脚本配置：

```bash
sed -n '1,240p' scripts/simulation/teleop_dual_ur5e_mujoco.py
```

---

## 19. 常见问题排查

### 19.1 PICO 已连接，但 Python 数据不变化

检查：

```text
Controller Tracking：ON
Send Tracking Data：ON
PC Service：正在运行
PICO 与电脑处于同一局域网
```

重新启动顺序：

```text
1. 关闭 PICO 中的 XRoboToolkit
2. 结束旧 PC Service
3. 重新启动 PC Service
4. 再打开 PICO XRoboToolkit
5. 重新连接电脑
```

### 19.2 PC Service 重复启动

检查：

```bash
pgrep -af "/opt/apps/roboticsservice"
```

结束旧进程后重新启动：

```bash
pkill -f "/opt/apps/roboticsservice"
/opt/apps/roboticsservice/runService.sh
```

### 19.3 找不到 xrobotoolkit_sdk

```bash
conda activate xr-robotics
cd ~/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python
bash setup_conda.sh --install
```

检查：

```bash
python -c "import xrobotoolkit_sdk; print(xrobotoolkit_sdk.__file__)"
```

### 19.4 MuJoCo 窗口打开，但机械臂不动

检查：

```text
PICO 是否为 WORKING
手柄位姿是否持续变化
Grip 是否能达到接近 1.0
是否一直按住对应 Grip
```

先用官方连续打印程序确认：

```bash
python dependencies/XRoboToolkit-PC-Service-Pybind/examples/run_binding_continuous.py
```

### 19.5 机械臂运动过快

先在较小范围内缓慢移动手柄。

如果脚本支持命令行比例参数，可以尝试：

```bash
python scripts/simulation/teleop_dual_ur5e_mujoco.py \
    --scale-factor 0.5
```

如果脚本不接受该参数，则需要查看脚本中的 `scale_factor` 默认值后再修改。

### 19.6 Hand State 全部相同

如果没有启用裸手追踪，Hand State 可能是默认占位值。这不会影响基于手柄的双 UR5e 遥操作。

### 19.7 Motion Tracker 数量为 0

```text
Number of Motion Trackers: 0
```

表示当前没有连接额外的 PICO Motion Tracker，对双 UR5e 手柄遥操作没有影响。

---

## 20. 完整验收清单

### PICO 与网络

```text
[ ] PICO 开发者模式已开启
[ ] USB 调试已开启
[ ] XRoboToolkit APK 已安装
[ ] PICO 与 Ubuntu 位于同一局域网
[ ] PICO 显示 PC connection established
[ ] XRoboToolkit 状态为 WORKING
```

### PC Service

```text
[ ] roboticsservice 安装成功
[ ] /opt/apps/roboticsservice/runService.sh 可执行
[ ] PC Service 进程正在运行
[ ] 没有重复启动多个实例
```

### Python Binding

```text
[ ] xrobotoolkit_sdk 可以导入
[ ] 左右手柄位姿持续更新
[ ] 头显位姿持续更新
[ ] Grip、Trigger、摇杆和按键均能正确读取
[ ] 时间戳持续更新
```

### 双 UR5e Demo

```text
[ ] MuJoCo 双 UR5e 模型正常加载
[ ] 不按 Grip 时机械臂保持静止
[ ] 左 Grip 只激活左机械臂
[ ] 右 Grip 只激活右机械臂
[ ] 手柄平移能够驱动末端平移
[ ] 手柄旋转能够驱动末端旋转
[ ] 松开 Grip 后对应机械臂停止跟随
[ ] 再次按下 Grip 时重新建立参考零点
```

全部完成后，说明以下链路已经复现成功：

```text
PICO 4 Ultra
→ XRoboToolkit App
→ Wi-Fi
→ XRoboToolkit PC Service
→ Python Binding
→ 遥操作映射与逆运动学
→ MuJoCo 双 UR5e
```

---

## 21. 常用命令汇总

### 启动 PC Service

```bash
/opt/apps/roboticsservice/runService.sh
```

### 查看 PC Service 进程

```bash
pgrep -af "/opt/apps/roboticsservice"
```

### 进入环境

```bash
conda activate xr-robotics
cd ~/XRoboToolkit/XRoboToolkit-Teleop-Sample-Python
```

### 查看 PICO 原始数据

```bash
python dependencies/XRoboToolkit-PC-Service-Pybind/examples/run_binding_continuous.py
```

### 启动双 UR5e Demo

```bash
python scripts/simulation/teleop_dual_ur5e_mujoco.py
```

### 查看全部仿真 Demo

```bash
find scripts/simulation \
    -maxdepth 1 \
    -type f \
    -name "*.py" \
    -printf "%f\n" \
    | sort
```
