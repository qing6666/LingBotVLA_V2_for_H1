# LingBot-VLA 2.0 模型架构图

> **查看方式**：
> - VSCode：装扩展 `Markdown Preview Mermaid Support`，打开本文件预览
> - 在线：复制下方代码到 [mermaid.live](https://mermaid.live) 渲染 + 导出 PNG/SVG

---

## Mermaid 源码

```mermaid
flowchart TD
    %% ===== 样式定义 =====
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1
    classDef vlm fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef expert fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20
    classDef flow fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100
    classDef loss fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#b71c1c
    classDef output fill:#e0f7fa,stroke:#00838f,stroke-width:2px,color:#006064
    classDef config fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#333
    classDef preproc fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#f57f17

    %% ===== ① 输入层 =====
    subgraph IN["📥 输入层"]
        I1["📷 3×RGB 相机<br/>camera_top / wrist_left / wrist_right"]:::input
        I2["🗣️ 任务语言<br/>lang_tokens"]:::input
        I3["🦾 关节状态 20维<br/>双臂16 各7关节+1夹爪<br/>头2 俯仰+旋转<br/>腰2 俯仰+旋转"]:::input
    end

    %% ===== 数据预处理 =====
    PRE["⚙️ robot_config 映射 + norm_stats 归一化<br/>→ padding 到 55 维统一空间"]:::preproc

    %% ===== ② VLM 骨干 =====
    subgraph VLM["🧠 VLM 骨干 — QwenvlWithExpertV2Model (Qwen3-VL 改装版)"]
        direction TB
        V1["视觉编码器<br/>3×RGB → patch_embed → VisionBlock<br/>cu_seqlens 分段注意力 → merger"]:::vlm
        V2["★ 双查询插入 Dual-Query<br/>depth query (深度提问位)<br/>video query (未来提问位)"]:::vlm
        V3["语言模型 Decoder<br/>prefix + suffix 联合注意力<br/>→ outputs_embeds + suffix_out"]:::vlm
    end

    %% ===== ③ MoE 动作专家 =====
    subgraph MOE["⚙️ MoE 动作专家 — 32选4 (qwen2_action_expert.py)"]
        direction TB
        E1["路由器 gate<br/>Linear fp32 打分<br/>+ e_score_correction_bias (loss-free 均衡)"]:::expert
        E2["top_k = 4 → 32专家选4<br/>Qwen2FusedExperts (3D权重 + fused_moe)"]:::expert
        E3["共享专家 shared expert<br/>所有 token 都过<br/>提取通用知识"]:::expert
        E4["action_out_proj<br/>→ v_t (预测速度)"]:::expert
    end

    %% ===== ④ Flow Matching =====
    subgraph FM["🌊 Flow Matching 动作生成 (modeling_lingbot_vla_v2.py + flow_match.py)"]
        direction LR
        F1["🟢 训练 forward<br/>noise~N(0,I), time~U(0,1)<br/>x_t = t·noise + 1-t ·action<br/>u_t = noise - action<br/>vla_loss = MSE(u_t, v_t)"]:::flow
        F2["🟧 推理 sample_actions<br/>1. prefix前向→缓存KV<br/>2. x_t=纯噪声 time=1.0<br/>3. 循环10步去噪:<br/>   v_t=predict_velocity<br/>   x_t += dt·v_t<br/>4. → 动作chunk(50步)"]:::flow
    end

    %% ===== ⑤ 三支线蒸馏 =====
    subgraph DIST["🎯 双查询蒸馏 — 训练时 3 支线并行 (推理只用动作)"]
        direction LR
        D1["① 动作主线<br/>v_t vs u_t<br/><b>vla_loss × 1.0</b>"]:::loss
        D2["② 深度蒸馏<br/>depth query vs MoGe+MoRGBD教师<br/>(教师从RGB估深度)<br/><b>depth_loss × 0.004</b>"]:::loss
        D3["③ 视频蒸馏<br/>video query vs DINO教师<br/>(教师从时序推未来)<br/><b>video_loss × 0.004</b>"]:::loss
    end

    %% ===== ⑥ 输出 =====
    subgraph OUT["📤 输出"]
        O1["训练: total_loss → backward"]:::output
        O2["推理: 动作chunk → unapply反归一化 → 机器人执行"]:::output
    end

    %% ===== ⑦ 配置支撑 =====
    subgraph CFG["⚙️ 支撑层：配置 + 数据管线"]
        direction LR
        C1["LingbotVLAV2Config<br/>模型设计图纸<br/>(32专家/10步去噪/align_params)"]:::config
        C2["robot_config.yaml<br/>字段翻译规则<br/>(12维→arm+effector)"]:::config
        C3["norm_stats.json<br/>归一化统计量<br/>(mean/std/q01/q99)"]:::config
    end

    %% ===== 主线连接 =====
    I1 --> PRE
    I2 --> PRE
    I3 --> PRE
    PRE --> V1
    V1 --> V2
    V2 --> V3

    V3 -->|"suffix_out"| E1
    E1 --> E2
    E2 --> E4
    E3 --> E4
    E4 -->|"v_t"| F1

    %% 蒸馏支线
    F1 --> D1
    V3 -.->|"depth query 输出"| D2
    V3 -.->|"video query 输出"| D3

    %% 汇总 loss
    D1 --> O1
    D2 --> O1
    D3 --> O1

    %% 推理路径
    V3 -->|"suffix_out"| F2
    F2 --> O2

    %% 配置支撑（虚线）
    C1 -.->|"配置"| VLM
    C1 -.->|"配置"| MOE
    C1 -.->|"配置"| FM
    C2 -.->|"映射"| PRE
    C3 -.->|"归一化"| PRE
```

---

## 架构要点

| 层 | 组件 | 核心作用 |
|---|---|---|
| ① 输入 | 3×RGB + 语言 + 状态 | 推理只需这 3 样(无深度/未来) |
| ② VLM 骨干 | Qwen3-VL 改装 | 看图听话 + 插入 depth/video 双查询 |
| ③ MoE 专家 | 32选4 + 共享专家 | 稀疏推理,容量大算得少 |
| ④ Flow Matching | 速度场学习 | 训练学方向,推理从噪声走到动作 |
| ⑤ 蒸馏 | 深度+视频教师 | 训练强化表征,推理只用动作受益 |
| ⑥ 输出 | loss / 动作chunk | 训练 backward;推理 unapply 还原 |
| ⑦ 配置 | Config+robot_config+norm | 设计图纸+翻译+归一化 |
