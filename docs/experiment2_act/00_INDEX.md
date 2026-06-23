# ACT 跨环境泛化实验 — 文档索引

本目录包含完整实验报告的分章节文档。建议按编号顺序阅读；`report.md` 为面向提交的摘要版。

## 文档结构

| 编号 | 文件 | 内容 |
|------|------|------|
| 00 | [00_INDEX.md](00_INDEX.md) | 本索引 |
| 01 | [01_background_ACT_CALVIN_LeRobot.md](01_background_ACT_CALVIN_LeRobot.md) | 背景：机器人模仿学习、ACT、CALVIN、LeRobot |
| 02 | [02_datasets_preprocessing.md](02_datasets_preprocessing.md) | 数据集划分、特征命名、合并与预处理 |
| 03 | [03_experimental_design.md](03_experimental_design.md) | 实验目的、变量、对照组、评估协议 |
| 04 | [04_training_analysis.md](04_training_analysis.md) | 训练过程、loss/梯度/吞吐、两模型对比 |
| 05 | [05_evaluation_generalization.md](05_evaluation_generalization.md) | 环境 D 零样本评估、统计表、误差分布 |
| 06 | [06_supplementary_experiments.md](06_supplementary_experiments.md) | 补充实验：checkpoint 消融、action chunk 视界 |
| 07 | [07_discussion_conclusion.md](07_discussion_conclusion.md) | 讨论、局限、结论与未来工作 |
| 08 | [08_timing_and_cost.md](08_timing_and_cost.md) | **训练/评估/总耗时**、资源与小型实验预算 |
| — | [report.md](report.md) | 实验报告摘要（可提交版骨架） |

## 图表与数据

| 类型 | 路径 |
|------|------|
| 全部 figures | `doc/figures/` |
| CSV 统计表 | `doc/stats/` |
| JSON 汇总 | `doc/stats/experiment_summary.json` |
| 原始 eval | `outputs/eval_envD_results.json` |
| 补充 eval | `outputs/supplementary_eval.json` |
| 训练 log | `outputs/act_envA_run.log`, `outputs/act_envABC_run.log` |

## 复现

```bash
cd /home1/lixiang/ACT/work && conda activate lerobot
python code/train_envA.py          # 单环境训练
python code/train_envABC.py        # 多环境训练
python code/eval_envD.py           # 快速 eval
python code/analyze_experiments.py # 统计 + 补充实验 + 全部图表
python code/plot_curves.py         # 基础曲线（已被 analyze 覆盖）
```

## 图表清单（不重复）

1. `dataset_scale.png` — 各 split 规模
2. `training_dashboard.png` — 训练四联图（loss / grad / data time / throughput）
3. `training_loss_curves.png` / `training_grad_norm.png` / … — 单指标大图
4. `eval_medium_overall.png` — 中等规模零样本 L1 对比
5. `eval_medium_per_dim.png` — 分维度柱状图
6. `eval_delta_per_dim.png` — A+B+C 相对 A 的维度差
7. `eval_error_histogram.png` — 误差直方图
8. `eval_error_cdf.png` — 误差 CDF
9. `ablation_checkpoint_envA.png` — 500 vs 1000 step checkpoint
10. `chunk_horizon_l1.png` — chunk 内各步 L1 曲线
11. `chunk_horizon_heatmap.png` — 模型 × chunk index 热力图
12. `timing_training_timeline.png` — 训练各阶段 wall 时间分解
