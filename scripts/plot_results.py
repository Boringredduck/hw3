import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_loss_log(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        return [], []
    pattern = re.compile(r"step:(\d+).*?(?:l1_loss|loss):([\d.]+)")
    steps, values = [], []
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            values.append(float(match.group(2)))
    return steps, values


def smooth(values: list[float]) -> list[float]:
    if len(values) < 20:
        return values
    window = max(5, len(values) // 40)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)].tolist()


def plot_loss(work_dir: Path, out_dir: Path) -> None:
    series = [
        ("ACT env A", work_dir / "act_envA.log", "#2563eb"),
        ("ACT A+B+C", work_dir / "act_envABC.log", "#f97316"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plotted = False
    for label, path, color in series:
        steps, values = parse_loss_log(path)
        if not steps:
            continue
        ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(steps, smooth(values), color=color, linewidth=2.0, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Training step")
    ax.set_ylabel("Action L1 loss")
    ax.set_title("ACT training loss on CALVIN")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_comparison.png", dpi=180)
    plt.close(fig)


def normalize_eval(payload: dict) -> dict:
    rows = payload.get("results", payload)
    if isinstance(rows, dict):
        return rows
    return {
        item["model"]: {
            "mean": item["mean_l1_total"],
            "std": item["std_l1_total"],
            "per_dim": item.get("per_dim_mean_l1", []),
            "dim_names": item.get("dim_names", []),
        }
        for item in rows
    }


def plot_eval(eval_path: Path, out_dir: Path) -> None:
    if not eval_path.exists():
        return
    results = normalize_eval(json.loads(eval_path.read_text()))
    keys = [key for key in ["act_envA", "act_envABC"] if key in results]
    if not keys:
        return
    labels = ["ACT env A" if key == "act_envA" else "ACT A+B+C" for key in keys]
    means = [results[key]["mean"] for key in keys]
    stds = [results[key]["std"] for key in keys]
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#2563eb", "#f97316"][: len(keys)], edgecolor="black")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.015, f"{mean:.3f}", ha="center")
    ax.set_ylabel("Mean Action L1")
    ax.set_title("Zero-shot Action L1 on Env D")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_envD_overall.png", dpi=180)
    plt.close(fig)

    if len(keys) == 2 and results[keys[0]]["per_dim"]:
        dim_names = results[keys[0]].get("dim_names") or [f"d{i}" for i in range(len(results[keys[0]]["per_dim"]))]
        x = np.arange(len(dim_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x - width / 2, results[keys[0]]["per_dim"], width, label=labels[0], color="#2563eb")
        ax.bar(x + width / 2, results[keys[1]]["per_dim"], width, label=labels[1], color="#f97316")
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names)
        ax.set_ylabel("Mean Action L1")
        ax.set_title("Per-dimension Action L1 on Env D")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "eval_envD_per_dim.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--out-dir", default="docs/experiment2_act/figures")
    parser.add_argument("--eval-json", default="")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = Path(args.eval_json).expanduser().resolve() if args.eval_json else work_dir / "eval_envD_results.json"

    plot_loss(work_dir, out_dir)
    plot_eval(eval_path, out_dir)
    print(f"figures saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
