# 08 实验耗时与资源统计

## 8.1 设计原则：最小化但高价值

本实验刻意采用 **「小型快速」** 预算，但每个环节都有独立信息量，避免重复劳动：

| 环节 | 最小化策略 | 保留的价值 |
|------|------------|------------|
| 训练 | 1000 steps（非 50k） | 完整 loss 下降曲线 + 双 checkpoint |
| 快速 eval | 10 ep × 20 batch | pipeline 验证、两模型可比 |
| 补充 eval | 20 ep × 25 batch 等 | 更稳均值 + 分布/CDF |
| Checkpoint 消融 | 仅 envA 500 vs 1000 | 训练进度对泛化的影响 |
| Chunk 视界 | 7 个 index 点 | ACT 特有，非重复柱状图 |

> 原则：**少而精** — 不做全量 30 万帧 eval（数小时），但做 **分布、视界、checkpoint** 三类不可替代分析。

---

## 8.2 训练阶段耗时（Wall-Clock）

数据来源：`outputs/act_envA_run.log`、`outputs/act_envABC_run.log`  
汇总 CSV：`doc/stats/timing_summary.csv`

![Training timeline](figures/timing_training_timeline.png)

| 模型 | 数据集加载 | 策略初始化 | 训练 1000 step | **总 Wall 时间** | **秒/step** |
|------|------------|------------|----------------|------------------|-------------|
| ACT env A | ~0s | ~2s | **~9m 03s** | **~9m 11s** | ~0.54s |
| ACT A+B+C | **~12m 39s** | ~2s | **~11m 51s** | **~24m 35s** | ~0.71s |

**关键观察**：
1. **merged 数据集加载** 占 envABC 总时间一半以上（17870 episodes 索引），这是多环境训练的真实工程成本。
2. 进入训练后，envABC 每 step 更慢（`data_s` 更大、吞吐 ~12 vs ~16 smp/s），因单 step 需从更大数据集采样。
3. envA 在 500 step 附近有一次 dataloader 抖动（log 中 step 501 附近 progress bar 跳变），导致前后半段各 ~4.3–4.5 min。

### 8.2.1 训练吞吐（从 log 统计）

| 模型 | 平均 `data_s`/step | 平均 `smp/s` | 峰值 GPU 内存 |
|------|-------------------|--------------|---------------|
| env A | ~0.43s | ~16 | ~1.03 GB |
| A+B+C | ~0.60s | ~12 | ~1.04 GB |

---

## 8.3 评估阶段耗时

### 8.3.1 快速 eval（`eval_envD.py` 默认）

配置：10 episodes × 20 batches × batch 16 ≈ **320 样本/模型**

| 阶段 | 耗时 |
|------|------|
| act_envA（load + infer） | ~1m 34s |
| act_envABC | ~1m 29s |
| **合计** | **~3m 03s** |

瓶颈：LeRobotDataset 构建 + 双相机 ResNet forward；首模型 load ~6s，dataset ~6s。

### 8.3.2 补充 eval（`analyze_experiments.py`）— 实测

| 子实验 | 配置 | wall（实测） |
|--------|------|--------------|
| medium_eval ×2 | 20ep×25batch | 124s + 115s |
| checkpoint 消融 ×2 | 15ep×20batch | 101s + 89s |
| chunk 视界 ×2 | 12ep×18batch×7 index | ~180s×2（含在同一 run） |
| **补充实验合计** | — | **609s（10.2 min）** |

---

## 8.4 端到端实验总账（估算）

| 项目 | 耗时 |
|------|------|
| 数据预处理（normalize + merge，一次性） | ~10–30 min（历史） |
| 训练 envA | ~9 min |
| 训练 envABC | ~25 min |
| 快速 eval | ~3 min |
| 补充分析 + 出图 | ~20 min |
| **可复现 rerun 核心路径** | **~57 min** |

并行训练 envA 与 envABC 时，训练段 wall 可压至 ~25 min，**总计 ~50 min** 完成「训练 + eval + 分析」。

---

## 8.5 与「全量实验」对比

| 方案 | 训练 | Eval | 总时长量级 | 适用 |
|------|------|------|------------|------|
| **本实验（小型）** | 1k steps ×2 | ~400–500 样本/模型 | **~1h** | 作业 pipeline、趋势、分析框架 |
| 标准 | 50k steps ×2 | 5k ep splitD | **数天** | 论文级 success rate |
| 全量 eval only | — | 308k frames | **10h+** | 精确 L1 Population |

小型实验 **不能** 得出「多环境一定更好」的强结论，但 **能** 支撑：
- 训练动态差异（I/O、loss、梯度）
- 零样本 L1 点估计 + 分布形态
- Action chunk 误差随 horizon 增长（ACT 机制验证）
- Checkpoint 步数对泛化的单调性检查

---

## 8.6 复现耗时统计

```bash
python code/collect_timing.py      # 仅从 log 解析，<5s
python code/analyze_experiments.py # 补充实验+图表，~20min
```

输出：
- `doc/stats/timing_stats.json`
- `doc/stats/timing_summary.csv`
- `doc/figures/timing_training_timeline.png`
