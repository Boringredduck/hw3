# 03 实验设计

## 3.1 研究问题

> **在固定 ACT 架构与超参下，仅在环境 A 上训练 vs 在 A+B+C 混合数据上训练，哪个在未见环境 D 上 zero-shot action 预测更准？**

衍生问题：
1. 多环境训练是否带来更好的 cross-domain 表征，还是因数据分布变宽而 **牺牲单步拟合**？
2. Action chunking 的误差如何随 chunk 内 horizon 增长？
3. 训练步数不足时（1000 steps），checkpoint 早期与晚期的泛化差异？

---

## 3.2 自变量与对照

| 因素 | Baseline (M1) | Treatment (M2) | 控制 |
|------|---------------|----------------|------|
| 训练数据 | splitA only | merged A+B+C | 相同 ACT 配置 |
| 训练 steps | 1000 | 1000 | 相同 |
| batch_size | 8 | 8 | 相同 |
| seed | 1000 | 1000 | 相同 |
| GPU | CUDA:1 | CUDA:4 | 单卡 |
| 评估数据 | splitD | splitD | 相同 protocol |

**不改变的**：网络结构、chunk_size=100、优化器/lr、预处理器、eval 脚本。

---

## 3.3 超参数表（完整）

| 参数 | 值 | 说明 |
|------|-----|------|
| `policy.type` | act | Action Chunking Transformer |
| `chunk_size` | 100 | 预测 horizon |
| `n_action_steps` | 100 | 与 chunk 一致 |
| `n_obs_steps` | 1 | 单帧观测 |
| `dim_model` | 512 | Transformer 宽度 |
| `n_heads` | 8 | 注意力头数 |
| `dim_feedforward` | 3200 | FFN 维度 |
| `n_encoder_layers` | 4 | 编码层 |
| `n_decoder_layers` | 1 | 解码层 |
| `use_vae` | true | CVAE 分支 |
| `latent_dim` | 32 | |
| `kl_weight` | 10.0 | KL 系数 |
| `vision_backbone` | resnet18 | 双相机共享 backbone 逻辑 |
| `batch_size` | 8 | 有效 batch |
| `steps` | 1000 | 快速完整 pipeline |
| `save_freq` | 500 | checkpoint @ 500, 1000 |
| `log_freq` | 100 | 每 100 step 记录 |
| `num_workers` | 4 | DataLoader |
| `lr` | 1e-5 | LeRobot 默认 |
| `seed` | 1000 | 可复现 |

脚本入口：`code/train_envA.py`、`code/train_envABC.py`。

---

## 3.4 评估协议

### 3.4.1 主评估：环境 D zero-shot L1

- **模型输入**：splitD 的 (image, wrist, state) @ t
- **模型输出**：action chunk â_{t:t+H}
- **指标**：归一化空间 L1，默认比较 **chunk 第 0 步** â_t 与 a_t

**两档评估规模**：

| 档位 | episodes | batches | 约样本数 | 用途 |
|------|----------|---------|----------|------|
| 快速 | 10 | 20 | ~320 | pipeline 冒烟 |
| **中等（报告主表）** | 30 | 40 | ~640 | 统计更稳 |

### 3.4.2 为何不用 success rate？

CALVIN 官方评估需 **语言条件 + 仿真环境 + 长任务链**。本仓库聚焦 LeRobot 离线训练 pipeline，**未部署 CALVIN sim**。因此：

- 使用 **held-out splitD 上的 action L1** 作为可复现、可自动化的 proxy
- 在讨论中明确：L1 ↓ 不等价于 success ↑，但二者通常相关

---

## 3.5 补充实验设计

| 编号 | 名称 | 目的 |
|------|------|------|
| E1 | Checkpoint 消融 (envA) | step 500 vs 1000 在 D 上 L1 |
| E2 | Chunk horizon 曲线 | index ∈ {0,1,5,10,25,50,99} 的 L1 |
| E3 | 误差分布 / CDF | 看 tail error、稳定性 |
| E4 | 分维度 Δ | A+B+C 相对 A 各 DoF 差异 |

执行：`python code/analyze_experiments.py`

---

## 3.6 实验流程时间线

1. 环境：`conda env lerobot`，PyTorch + CUDA
2. 数据：normalize keys → merge ABC
3. 训练：envA ∥ envABC（各 1000 steps，~9–12 min / 25 min）
4. Eval：`eval_envD.py` + `analyze_experiments.py`
5. 图表与文档：`doc/figures/`, `doc/stats/`, 分章节 md

---

## 3.7 公平性与已知 confound

| 因素 | 影响 |
|------|------|
| 1000 steps 极短 | 两模型均未收敛；结论为 **趋势演示** 非 SOTA |
| merged 数据加载更慢 | envABC 每 step `data_s` 更大，相同样本数下「见过」的 unique 帧更少 |
| 相同 seed | 不保证两 run 覆盖相同 shuffle 轨迹比例 |
| L1 on normalized action | 不同维度量纲已归一，但仍可能掩盖 gripper 等关键维 |

这些在 [07_discussion_conclusion.md](07_discussion_conclusion.md) 展开。
