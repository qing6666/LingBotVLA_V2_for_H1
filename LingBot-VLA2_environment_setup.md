# LingBot-VLA 2.0 环境配置教程

## 1. 项目简介

LingBot-VLA 2.0 是一个
Vision-Language-Action（VLA）基础模型项目，用于将视觉、语言指令和机器人动作进行统一建模。

本教程记录从零开始配置 LingBot-VLA 2.0 开发环境的流程。

------------------------------------------------------------------------

# 2. 硬件环境建议

## 推荐配置

  项目   推荐
  ------ ---------------------------
  GPU    RTX4090 / RTX5090 / A6000
  显存   24GB以上更佳
  CPU    16核以上
  内存   64GB以上
  硬盘   200GB以上

## RTX5070环境说明

RTX5070（12GB显存）：

可以用于：

-   项目环境搭建
-   源码学习
-   数据处理
-   小规模推理
-   量化模型推理尝试

不适合：

-   LingBot-VLA 2.0 全量训练
-   大规模预训练

------------------------------------------------------------------------

# 3. 软件环境要求

LingBot-VLA 2.0 官方要求：

-   Ubuntu Linux
-   Miniconda / Anaconda
-   Python 3.12
-   PyTorch 2.8.0

------------------------------------------------------------------------

# 4. 安装Conda环境

## 4.1 检查Conda

``` bash
conda --version
```

如果没有安装：

建议安装 Miniconda。

------------------------------------------------------------------------

## 4.2 初始化Conda

``` bash
conda init bash
```

重新打开终端。

测试：

``` bash
conda activate
```

如果可以正常进入环境，说明初始化完成。

------------------------------------------------------------------------

# 5. 下载项目源码

进入工作目录：

``` bash
cd ~/robot_item
```

下载：

``` bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
```

进入项目：

``` bash
cd lingbot-vla-v2
```

------------------------------------------------------------------------

# 6. 创建LingBot-VLA环境

官方提供自动安装脚本：

``` bash
bash tools/create_train_env.sh
```

该脚本会自动安装：

-   Python环境
-   PyTorch
-   依赖库
-   Flash Attention

------------------------------------------------------------------------

## 指定环境名称

例如：

``` bash
bash tools/create_train_env.sh \
--env-name lingbotvla
```

进入环境：

``` bash
conda activate lingbotvla
```

------------------------------------------------------------------------

## 强制重新创建环境

如果安装失败：

``` bash
bash tools/create_train_env.sh \
--env-name lingbotvla \
--recreate
```

------------------------------------------------------------------------

# 7. 验证PyTorch环境

进入Python：

``` bash
python
```

执行：

``` python
import torch

print(torch.__version__)

print(torch.cuda.is_available())

print(torch.version.cuda)
```

正常情况：

应该看到：

-   PyTorch 2.8.x
-   CUDA可用
-   GPU可以识别

------------------------------------------------------------------------

# 8. 验证GPU

终端：

``` bash
nvidia-smi
```

确认：

-   GPU型号
-   驱动版本
-   显存大小

------------------------------------------------------------------------

# 9. Flash Attention检查

执行：

``` bash
python -c "import flash_attn; print('flash attention ok')"
```

如果失败：

重新安装：

``` bash
pip install flash-attn==2.8.3
```

------------------------------------------------------------------------

# 10. 下载预训练模型

LingBot-VLA 2.0提供：

    lingbot-vla-v2-6b

下载：

``` bash
python scripts/download_hf_model.py \
--repo_id robbyant/lingbot-vla-v2-6b \
--local_dir lingbot-vla
```

下载后：

目录：

    lingbot-vla/
    |
    ├── model
    ├── config
    └── depth

------------------------------------------------------------------------

# 11. 第一次运行建议

不要直接训练。

推荐顺序：

    环境配置
        |
        ↓
    模型下载
        |
        ↓
    源码阅读
        |
        ↓
    推理测试
        |
        ↓
    数据准备
        |
        ↓
    微调训练

------------------------------------------------------------------------

# 12. 后续学习重点目录

## 训练入口

    train.sh

## 训练代码

    tasks/vla/train_lingbotvla.py

## 模型代码

    lingbotvla/

## 数据处理

    lingbotvla/data/vla_data/

## 机器人配置

    configs/robot_configs/

## 部署

    deploy/

------------------------------------------------------------------------

# 13. RTX5070推荐运行策略

RTX5070 12GB显存：

推荐：

-   FP16推理
-   INT4量化推理
-   CPU offload
-   LoRA/Adapter微调

不推荐：

-   全参数训练

------------------------------------------------------------------------

# 14. 下一步学习路线

    环境搭建
       |
       ↓
    理解train.sh
       |
       ↓
    分析train_lingbotvla.py
       |
       ↓
    分析模型结构
       |
       ↓
    理解55维Action
       |
       ↓
    适配自己的机器人
       |
       ↓
    真实机器人部署

------------------------------------------------------------------------

# 备注

本文档用于个人学习记录，后续将继续补充：

-   LingBot-VLA代码解析
-   数据集制作
-   自定义机器人适配
-   SO101/自研机器人部署流程
