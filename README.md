# HW3 Deep Learning and Spatial Intelligence

本仓库整理了《深度学习与空间智能》期末作业的两个实验方向：

- 题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合
- 题目二：基于 LeRobot 的 ACT 策略跨环境泛化挑战

最终报告位于 `docs/final_report/report.pdf`。LaTeX 源文件和报告图片也保留在 `docs/final_report/`，可重新编译。

## Repository Structure

```text
.
├── README.md
├── environment.yml
├── requirements.txt
├── scripts/
│   ├── download_calvin.py
│   ├── prepare_calvin.py
│   ├── train_act.py
│   ├── eval_envd.py
│   └── plot_results.py
└── docs/
    ├── final_report/
    │   ├── report.pdf
    │   ├── report.tex
    │   └── act_figures/
    ├── experiment1_3d_assets/
    │   ├── images/
    │   ├── models/
    │   └── videos/
    └── experiment2_act/
        ├── figures/
        ├── stats/
        ├── results/
        └── report.md
```

`data/`、`runs/`、checkpoint、wandb 日志、模型权重和训练/验证/测试数据不提交到 GitHub。需要复现实验时按下面步骤在本地生成。

材料目录说明：

- `docs/final_report/`：最终报告 PDF、LaTeX 源文件和报告所需图片。
- `docs/experiment1_3d_assets/`：题目一的交付模型、截图和渲染视频。
- `docs/experiment2_act/`：题目二的实验记录、图表、统计文件和小型评估 JSON。
- `scripts/`：题目二的数据准备、训练、测试和绘图脚本。

## Requirements

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate hw3-lerobot-act
```

如果已有 Python 环境，也可以直接安装依赖：

```bash
python -m pip install -r requirements.txt
```

主要依赖包括 PyTorch、LeRobot、Hugging Face Hub、NumPy、Pandas、PyArrow、Matplotlib 和 tqdm。ACT 训练建议使用 Linux + CUDA GPU 环境。

## Experiment 1: 3D Assets and Scene Fusion

题目一材料位于 `docs/experiment1_3d_assets/`：

- `images/`：输入图、重建/生成结果截图
- `models/`：扑克牌、鼠标、充电插头、garden 场景的 OBJ/PLY 交付模型
- `videos/`：单物体环绕视频与 Blender 最终融合视频

主要流程包括：

1. 使用 COLMAP + 3DGS 从扑克牌多视角视频重建点云和模型。
2. 使用文本到 3D 方法生成黑色无线鼠标模型。
3. 使用单图到 3D 方法生成充电插头模型。
4. 使用 3DGS 重建 garden 背景场景。
5. 在 Blender 中统一尺度、位置和光照，完成多资产融合渲染。

题目一的可交付结果已经整理为模型、图片和视频文件；原始训练数据和外部工具工程目录未纳入仓库。

## Experiment 2: LeRobot ACT on CALVIN

题目二代码位于 `scripts/`，实验文档、图表和统计结果位于 `docs/experiment2_act/`。

### Data Preparation

下载 CALVIN LeRobot 数据集：

```bash
python scripts/download_calvin.py \
  --repo-id xiaoma26/calvin-lerobot \
  --output-dir data/calvin-lerobot
```

转换数据格式、统一 LeRobot feature key，并合并 A+B+C：

```bash
python scripts/prepare_calvin.py all \
  --data-root data/calvin-lerobot \
  --merged-root data/merged_ABC \
  --repo-prefix calvin \
  --workers 4
```

如果数据已经是 LeRobot v3 格式，可只执行 key 归一化与合并：

```bash
python scripts/prepare_calvin.py normalize \
  --data-root data/calvin-lerobot \
  --repo-prefix calvin \
  --workers 4

python scripts/prepare_calvin.py merge \
  --data-root data/calvin-lerobot \
  --merged-root data/merged_ABC \
  --repo-prefix calvin \
  --workers 4
```

### Train

训练环境 A 单域 baseline：

```bash
python scripts/train_act.py \
  --variant envA \
  --data-root data/calvin-lerobot \
  --work-dir runs \
  --gpu 0 \
  --steps 1000 \
  --batch-size 8 \
  --num-workers 4 \
  --overwrite
```

训练 A+B+C 多环境模型：

```bash
python scripts/train_act.py \
  --variant envABC \
  --data-root data/calvin-lerobot \
  --merged-root data/merged_ABC \
  --work-dir runs \
  --gpu 0 \
  --steps 1000 \
  --batch-size 8 \
  --num-workers 4 \
  --overwrite
```

训练输出默认保存在 `runs/act_envA/` 和 `runs/act_envABC/`。这些目录包含 checkpoint 和日志，默认不进入 GitHub。

### Test

在未见环境 D 上做 zero-shot Action L1 测试：

```bash
python scripts/eval_envd.py \
  --data-root data/calvin-lerobot \
  --work-dir runs \
  --device cuda \
  --max-episodes 10 \
  --max-batches 20 \
  --batch-size 16 \
  --output runs/eval_envD_results.json
```

如果 checkpoint 不在默认位置，可显式指定：

```bash
python scripts/eval_envd.py \
  --data-root data/calvin-lerobot \
  --ckpt-env-a /path/to/act_envA/checkpoints/last/pretrained_model \
  --ckpt-env-abc /path/to/act_envABC/checkpoints/last/pretrained_model \
  --device cuda \
  --output runs/eval_envD_results.json
```

生成训练曲线和测试图：

```bash
python scripts/plot_results.py \
  --work-dir runs \
  --eval-json runs/eval_envD_results.json \
  --out-dir docs/experiment2_act/figures
```

## Reported Results

报告中的主要题目二结果保存在 `docs/experiment2_act/stats/` 和 `docs/experiment2_act/results/`。在 1000-step 快速训练预算下，环境 A 单域模型在环境 D 上的 Action L1 低于 A+B+C 多环境模型；报告中将其解释为多环境模型在短训练预算下仍处于欠训练状态，而不是多环境训练本身无效。

## Model Weights

模型权重和完整实验结果不提交到 GitHub，单独通过百度网盘提供：

- 文件：实验结果与模型权重
- 链接：https://pan.baidu.com/s/1Vxqe9ldj-p4FHISICwnSig
- 提取码：mfkd

## GitHub Storage Policy

仓库保留代码、最终报告、图表、统计文件、题目一交付模型和视频。以下内容不提交：

- CALVIN 训练/验证/测试数据
- ACT checkpoint、模型权重和 Hugging Face 权重目录
- wandb 运行目录
- 本地训练输出和临时缓存
- LaTeX 编译中间文件
