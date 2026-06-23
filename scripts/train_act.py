import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_train_args(args, dataset_root: Path, output_dir: Path, repo_id: str) -> list[str]:
    return [
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.use_imagenet_stats=false",
        "--policy.type=act",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.chunk_size}",
        "--policy.n_obs_steps=1",
        "--policy.dim_model=512",
        "--policy.n_heads=8",
        "--policy.dim_feedforward=3200",
        "--policy.n_encoder_layers=4",
        "--policy.n_decoder_layers=1",
        "--policy.use_vae=true",
        "--policy.latent_dim=32",
        "--policy.kl_weight=10.0",
        "--policy.vision_backbone=resnet18",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
        f"--output_dir={output_dir}",
        f"--job_name=act_{args.variant}",
        "--wandb.enable=true",
        f"--wandb.mode={args.wandb_mode}",
        "--wandb.project=act_calvin",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["envA", "envABC"], required=True)
    parser.add_argument("--data-root", default="data/calvin-lerobot")
    parser.add_argument("--merged-root", default="data/merged_ABC")
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--repo-prefix", default="calvin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    merged_root = Path(args.merged_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if args.variant == "envA":
        dataset_root = data_root / "splitA"
        repo_id = f"{args.repo_prefix}/splitA"
        output_dir = work_dir / "act_envA"
    else:
        dataset_root = merged_root
        repo_id = f"{args.repo_prefix}/mergedABC"
        output_dir = work_dir / "act_envABC"

    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"dataset not found: {dataset_root}")
    if args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)

    log_file = work_dir / f"act_{args.variant}.log"
    wandb_dir = work_dir / f"act_{args.variant}_wandb"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["WANDB_MODE"] = args.wandb_mode
    env["WANDB_DIR"] = str(wandb_dir)

    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    cmd.extend(build_train_args(args, dataset_root, output_dir, repo_id))

    print(" ".join(cmd))
    print(f"log: {log_file}")
    with log_file.open("w") as handle:
        return subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
