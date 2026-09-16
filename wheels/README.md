# wheels/ 说明

本目录原有 1 个文件,因超过 GitHub 单文件 100MB 上限未随仓库上传:

```
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl  (245MB)
```

## 它是什么

flash-attn 2.8.3 预编译 wheel,精确对应 **cp312 / torch 2.8.0 / cu12 / cxx11abi-TRUE** 组合
(本复现锁定的环境版本)。装错 ABI 的 wheel 会 import 失败,认准文件名。

## 怎么重新获取(三选一)

1. **官方 release 页**(推荐,文件名逐字对应):
   https://github.com/Dao-AILab/flash-attention/releases —— 找 `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
2. **源机器拷贝**(若是同一台机器的备份):本机 `lingbot_vla_v2_code_backup_2026-09-16.tar.gz`
   备份包内含此文件,解包 `wheels/` 即得
3. **pip 拉取**(版本一致但构建标签可能不同):
   `pip download flash_attn==2.8.3 --no-deps --python-version 3.12 --only-binary=:all:`

## 安装

```bash
pip install wheels/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
