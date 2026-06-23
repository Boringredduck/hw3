import argparse
import importlib
import json
import logging
from pathlib import Path

import numpy as np
import packaging
import torch
from torch.utils.data import DataLoader


if not hasattr(packaging, "version"):
    packaging.version = importlib.import_module("packaging.version")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_envd")

DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def read_fps(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return 10
    return int(json.loads(info_path.read_text()).get("fps", 10))


def load_policy(checkpoint: Path, device: str):
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.to(device)
    policy.eval()
    return policy


def build_dataset(policy_config, dataset_root: Path, repo_id: str, max_episodes: int | None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = read_fps(dataset_root)
    delta_timestamps = {
        "observation.images.image": [0.0],
        "observation.images.wrist_image": [0.0],
        "observation.state": [0.0],
        "action": [idx / fps for idx in range(policy_config.chunk_size)],
    }
    episodes = list(range(max_episodes)) if max_episodes is not None else None
    return LeRobotDataset(repo_id, root=str(dataset_root), episodes=episodes, delta_timestamps=delta_timestamps)


def to_device(batch: dict, device: str) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def normalize_images(batch: dict, camera_keys: list[str]) -> dict:
    batch = dict(batch)
    for key in camera_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            batch[key] = value.float() / 255.0
    return batch


def squeeze_observation_time(batch: dict, camera_keys: list[str]) -> dict:
    batch = dict(batch)
    for key in camera_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 5 and value.shape[1] == 1:
            batch[key] = value[:, 0]
    state = batch.get("observation.state")
    if isinstance(state, torch.Tensor) and state.ndim == 3 and state.shape[1] == 1:
        batch["observation.state"] = state[:, 0]
    return batch


@torch.no_grad()
def evaluate(policy, dataset, preprocessor, device: str, batch_size: int, max_batches: int | None, num_workers: int) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    errors = []
    camera_keys = dataset.meta.camera_keys
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        batch = normalize_images(batch, camera_keys)
        batch = preprocessor(batch)
        batch = squeeze_observation_time(batch, camera_keys)
        prediction = policy.predict_action_chunk(batch)[:, 0, :]
        target = batch["action"][:, 0, :]
        errors.append((prediction - target).abs().cpu().numpy())

    if not errors:
        raise RuntimeError("no evaluation samples were produced")

    error = np.concatenate(errors, axis=0)
    return {
        "n_samples": int(error.shape[0]),
        "action_dim": int(error.shape[1]),
        "mean_l1_total": float(error.mean()),
        "std_l1_total": float(error.std()),
        "per_dim_mean_l1": error.mean(axis=0).tolist(),
        "per_dim_std_l1": error.std(axis=0).tolist(),
        "dim_names": DIM_NAMES[: int(error.shape[1])],
    }


def run_one(name: str, checkpoint: Path, args) -> dict:
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    log.info("evaluating %s", name)
    policy = load_policy(checkpoint, args.device)
    policy.reset()
    dataset_root = Path(args.data_root).expanduser().resolve() / "splitD"
    dataset = build_dataset(policy.config, dataset_root, args.repo_id, args.max_episodes)
    preprocessor, _ = make_act_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    result = evaluate(policy, dataset, preprocessor, args.device, args.batch_size, args.max_batches, args.num_workers)
    result["model"] = name
    result["checkpoint"] = str(checkpoint)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/calvin-lerobot")
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--repo-id", default="calvin/splitD")
    parser.add_argument("--ckpt-env-a", default="")
    parser.add_argument("--ckpt-env-abc", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    checkpoints = {
        "act_envA": Path(args.ckpt_env_a or work_dir / "act_envA/checkpoints/last/pretrained_model"),
        "act_envABC": Path(args.ckpt_env_abc or work_dir / "act_envABC/checkpoints/last/pretrained_model"),
    }
    results = []
    for name, checkpoint in checkpoints.items():
        if checkpoint.exists():
            results.append(run_one(name, checkpoint, args))
        else:
            log.warning("missing checkpoint for %s: %s", name, checkpoint)

    payload = {
        "eval_mode": "envD_zero_shot_action_l1",
        "max_episodes": args.max_episodes,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "results": results,
    }
    output = Path(args.output or work_dir / "eval_envD_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{'model':<14} {'mean_l1':>10} {'std_l1':>10} {'samples':>8}")
    for item in results:
        print(f"{item['model']:<14} {item['mean_l1_total']:>10.4f} {item['std_l1_total']:>10.4f} {item['n_samples']:>8}")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
