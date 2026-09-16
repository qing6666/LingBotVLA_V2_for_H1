#!/home/mjq/miniconda3/envs/xr-robotics/bin/python
# -*- coding:utf-8 -*-

'''H1 仿真 —— PICO 纯遥操作启动器（不录数据）'''

"""
run_teleop.py
直接运行本文件 = 启动 PICO 遥操作（无录制）。

    ./run_teleop.py                # 直接执行（shebang 已指向 xr-robotics 环境）
    python run_teleop.py           # 等效（前提：当前是 xr-robotics 环境）
    ./run_teleop.py --debug-xr     # 额外参数原样透传给 teleop

内部等价于：
    conda activate xr-robotics
    cd H1_build && python teleop/h1_pico_teleop.py <透传参数>

前置：PC Service 已启动 + PICO 已连接（速查见 teleop/h1_pico_teleop.py 文末）。
"""

import os
import sys
from pathlib import Path

H1_ROOT = Path(__file__).resolve().parent                       # .../H1_build
PYTHON = "/home/mjq/miniconda3/envs/xr-robotics/bin/python"     # 遥操作/数采专用环境
TELEOP = H1_ROOT / "teleop" / "h1_pico_teleop.py"

if __name__ == "__main__":
    # execv 把当前进程替换成 teleop（工作目录无关，teleop 内部用 __file__ 定位）
    os.chdir(H1_ROOT)  # 数据/模型相对路径统一以 H1_build 为基准
    os.execv(PYTHON, [PYTHON, str(TELEOP), *sys.argv[1:]])
