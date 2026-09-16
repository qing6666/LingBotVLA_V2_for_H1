# B 方案补丁记录(2026-09-04)

## 改了什么(共 3 个核心文件)

- **补丁1:`lingbotvla/distributed/torch_parallelize.py`**(4 处)—— 放行"单卡 fsdp2 + offload"
- **补丁2:`tasks/vla/train_lingbotvla.py`**(~L965,新增 9 行)—— 首次试跑死于
  `clip_grad_norm_`:单卡 offload 下梯度是 CPU 上的 DTensor,torch 的裁剪会经 mesh 发
  NCCL 集合通信,NCCL 不支持 CPU → `No backend type associated with device type cpu`。
  修法:新增 `world_size == 1` 分支,对 `p.grad.to_local()`(1 卡 mesh 的 local=全量)
  直接裁剪,零通信。普通张量无 to_local 原样通过 → 非 offload 的单卡跑法行为不变。
- **补丁3:`lingbotvla/optim/muon.py`**(2 处,2026-09-04 10:33)—— 第二次试跑死于
  muon 优化器 `optimizer.step()`(muon.py:542):对 CPU 张量发 `dist.all_gather`,
  同样是 NCCL 不收 CPU。整条 muon 路径共两处集合通信,一并堵死:
  ① `_full_grad`:`full_tensor()` 内部隐式 all_gather → mesh 总尺寸==1 时 local 即全量,
  恒等返回(此一处覆盖 3 个调用点,含 3D MoE 专家张量那条道);
  ② `_step_megabatch_chunk` 显式 all_gather → `world_size == 1` 时直接 copy(数学恒等)。
  多卡(mesh>1)与非 offload 跑法(w 系列:普通张量,不进 DTensor 分支)逐比特不变。
- **补丁4:`lingbotvla/optim/muon.py`**(1 个包装函数 + 4 个调用点,2026-09-04 11:00)
  —— 第三次试跑没崩,但"卡住":offload 下 muon 在 CPU 上对 5.7B 参数做 Newton-Schulz,
  24 核全速要 10-20 分钟/步(20000 步≈208 天,速度死刑)。修法:新增
  `_newton_schulz_maybe_gpu` 包装——张量在 CPU 且有 GPU 时,搬上显卡算 NS(纯矩阵乘,
  5090 上 <2 秒,bf16 峰值 ~4G 显存)再搬回。同算法同精度,仅浮点最低位舍入随
  CPU/GPU 实现略有差异,对优化器语义无影响;张量本来在 GPU(多卡/无 offload)时
  直通零改动。预期每步 15-25 秒 → 20000 步 3-5 天。

框架其余文件零改动:parallel_state.py / deploy 全部未动。

目的:让"单卡 5090 + fsdp2 + `enable_fsdp_offload: true`"这条路能走通,跑全模型微调(B 方案)。
原代码单卡时 `fsdp_enabled` 恒 False → 守卫直接 raise offload,且 fsdp2 分支整段被跳过。

## 4 处改动(行号为补丁后)

1. **L45**:import 区加 `from torch.distributed.fsdp import CPUOffloadPolicy`
2. **L102-104**:函数开头预读 `enable_fsdp_offload` 标志(只 get 不 pop,fsdp1 分支仍自行 pop)
3. **L105**:守卫条件 `if kwargs.pop(...) and parallel_state.world_size > 1` —— 单卡不再 raise,多卡照旧
4. **L153**:大门 `if parallel_state.fsdp_enabled or enable_fsdp_offload:` —— 单卡+offload 也进入 fsdp2 分支
5. **L253-257**:fsdp2 分支内把 `CPUOffloadPolicy(pin_memory=True)` 挂到 `fsdp_kwargs` 和
   `mp_fsdp_kwargs` 的 `offload_policy`(fully_shard 官方参数名)

## 行为不变性(谁不受影响)

- 单卡、不开 offload(w 系列全部历史跑法):预读得 False,大门条件不变,**逐比特原行为**
- 多卡 fsdp1:守卫块不进入,分支内自行 pop,不变
- 多卡 fsdp2、不开 offload:仅多一次无害的 kwargs.get,不变
- 多卡 fsdp2、开 offload:新能力(原代码 fsdp2 分支从未消费过该标志)——本机从未跑过此组合

## 复原(方案 B 不行时,三条命令,全部恢复原样)

```bash
cp mjq_lingbotvla_v2/B_patch/torch_parallelize.py.orig lingbotvla/distributed/torch_parallelize.py
cp mjq_lingbotvla_v2/B_patch/train_lingbotvla.py.orig tasks/vla/train_lingbotvla.py
cp mjq_lingbotvla_v2/B_patch/muon.py.orig lingbotvla/optim/muon.py
```

复原后自检(三条 diff 均应无输出):

```bash
diff mjq_lingbotvla_v2/B_patch/torch_parallelize.py.orig lingbotvla/distributed/torch_parallelize.py && \
diff mjq_lingbotvla_v2/B_patch/train_lingbotvla.py.orig tasks/vla/train_lingbotvla.py && \
diff mjq_lingbotvla_v2/B_patch/muon.py.orig lingbotvla/optim/muon.py && echo 已全部复原
```

## 本目录文件

- `torch_parallelize.py.orig` —— 补丁1前原件(2026-09-04 10:09 备份,25757 字节)
- `torch_parallelize_offload.patch` —— 补丁1 unified diff(51 行)
- `train_lingbotvla.py.orig` —— 补丁2前原件(2026-09-04 10:25 备份)
- `train_lingbotvla_offload.patch` —— 补丁2 unified diff
- `muon.py.orig` —— 补丁3/4前原件(2026-09-04 10:33 备份,24389 字节)
- `muon_offload.patch` —— 补丁3+4 累积 unified diff
