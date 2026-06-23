# 01 背景知识：模仿学习、ACT、CALVIN 与 LeRobot

## 1.1 问题定义：机器人模仿学习

**模仿学习（Imitation Learning, IL）** 从人类示教数据中学习策略 π(a|s)，使机器人在相似状态下复现合理动作。与强化学习不同，IL 不依赖在线环境奖励，适合 CALVIN 这类长 horizon 操作任务。

典型流程：

1. **示教采集**：遥操作或 scripted demo → 轨迹 τ = {(o_t, a_t)}
2. **离线训练**：最小化预测动作与示教动作的偏差（如 L1 / MSE）
3. **部署评估**：在仿真或真机上 rollout，统计 task success rate

本实验处于第 2–3 步之间：完成离线训练，并用 **held-out 环境上的 action L1** 作为 zero-shot 泛化 proxy（无在线仿真 rollout）。

---

## 1.2 ACT（Action Chunking with Transformers）

**ACT**（Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023）是面向精细操作的模仿学习策略，核心思想：

| 组件 | 作用 |
|------|------|
| **CVAE 编码器** | 将 action 序列编码为 latent z，训练时提供序列级监督 |
| **ResNet 视觉 backbone** | 从第三人称 + 腕部相机提取特征 |
| **Transformer 编解码** | 融合图像 token、robot state、latent，解码 action chunk |
| **Action Chunking** | 一次预测 H 步未来动作（本实验 H=100），执行时可只用首步或整段开环 |

**为何用 chunking？** 单步 greedy 预测易抖动；chunk 建模短期时序，更平滑。代价是开环执行时误差随 horizon 累积——本报告第 06 章用 **chunk index 消融** 量化这一点。

**损失函数（训练）**：归一化 action 空间上的 L1 + KL（VAE 分支），与 LeRobot 默认 ACT 实现一致。

---

## 1.3 CALVIN 与 A/B/C/D 环境划分

**CALVIN**（Composing Actions from Language and Vision）是语言条件的长 horizon 桌面操作 benchmark。本作业使用其 **LeRobot 格式子集**，按环境 visual domain 划分为：

- **Env A / B / C / D**：四套不同场景纹理、光照、物体布局（domain shift）
- **训练**：仅在 A（baseline）或 A+B+C（multi-env）上训练
- **评估**：在 **从未见过的 D** 上 zero-shot 测试泛化

这对应经典 **domain generalization** 设定：训练域 ≠ 测试域，考察视觉与动力学表征是否足够 invariant。

---

## 1.4 LeRobot 框架

**LeRobot**（Hugging Face）提供：

- 统一 **LeRobotDataset**（parquet + mp4，delta_timestamps 对齐多帧 action chunk）
- **ACTPolicy** + preprocessor（归一化、device、batch 维）
- **lerobot_train** CLI / Python API

本实验代码位于 `work/code/`，数据在 `calvin-lerobot/split*` 与 `work/data/merged_ABC`。

### 关键 feature 命名（本实验统一后）

| Key | 含义 |
|-----|------|
| `observation.images.image` | 第三人称 RGB (200×200) |
| `observation.images.wrist_image` | 腕部 RGB (84×84) |
| `observation.state` | 机器人 proprio state |
| `action` | 7-DoF 末端 + gripper（归一化） |

LeRobot 的 `batch_to_transition` **仅保留 `observation.*` 前缀**；动作键必须为 `action`。详见 [02_datasets_preprocessing.md](02_datasets_preprocessing.md)。

---

## 1.5 本实验在文献/作业中的位置

| 维度 | 本实验 |
|------|--------|
| 策略 | ACT（固定架构） |
| 训练数据 | splitA vs merged A+B+C |
| 测试 | splitD zero-shot |
| 指标 | 归一化 action L1（proxy）；无 success rate |
| 训练预算 | 快速演示 1000 steps（可扩展至 50k+） |

**阅读后你应该知道**：我们比较的是「单域过拟合 baseline」与「多域混合训练」在未见域 D 上的 action 预测误差，而非 CALVIN 官方 language-conditioned success rate。
