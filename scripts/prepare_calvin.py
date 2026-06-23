import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_calvin")

FEATURE_ALIASES = {
    "image": "observation.images.image",
    "observation.image": "observation.images.image",
    "observation.images.image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "observation.wrist_image": "observation.images.wrist_image",
    "observation.images.wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "observation.state": "observation.state",
    "actions": "action",
    "action": "action",
}

IGNORED_DATA_COLUMNS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "source_frame_index",
    "source_episode_index",
}


def canonical(name: str) -> str:
    return FEATURE_ALIASES.get(name, name)


def canonical_stat_column(name: str) -> str:
    if not name.startswith("stats/"):
        return name
    parts = name.split("/")
    if len(parts) < 3:
        return name
    feature = "/".join(parts[1:-1])
    return f"stats/{canonical(feature)}/{parts[-1]}"


def rewrite_json_keys(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if path.name == "info.json":
        features = payload.get("features", {})
        new_features = {canonical(key): value for key, value in features.items()}
        changed = new_features != features
        if changed:
            payload["features"] = new_features
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return changed
    new_payload = {canonical(key): value for key, value in payload.items()}
    if new_payload != payload:
        path.write_text(json.dumps(new_payload, indent=2, ensure_ascii=False))
        return True
    return False


def rewrite_data_parquet(path: Path) -> bool:
    table = pq.read_table(path)
    names = [canonical(name) for name in table.column_names]
    if names == table.column_names:
        return False
    pq.write_table(table.rename_columns(names), path, compression="snappy")
    return True


def rewrite_episode_parquet(path: Path) -> bool:
    table = pq.read_table(path)
    names = [canonical_stat_column(name) for name in table.column_names]
    if names == table.column_names:
        return False
    pq.write_table(table.rename_columns(names), path, compression="snappy")
    return True


def run_pool(paths: list[Path], fn, label: str, workers: int) -> None:
    if not paths:
        return
    changed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, path): path for path in paths}
        for idx, future in enumerate(as_completed(futures), 1):
            if future.result():
                changed += 1
            if idx % 50 == 0:
                log.info("%s %d/%d", label, idx, len(paths))
    log.info("%s files=%d changed=%d", label, len(paths), changed)


def normalize_dataset(root: Path, workers: int) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        log.warning("skip missing dataset: %s", root)
        return
    log.info("normalizing %s", root)
    rewrite_json_keys(info_path)
    rewrite_json_keys(root / "meta" / "stats.json")
    run_pool(sorted((root / "data").rglob("*.parquet")), rewrite_data_parquet, "data", workers)
    episode_dir = root / "meta" / "episodes"
    if episode_dir.exists():
        run_pool(sorted(episode_dir.rglob("*.parquet")), rewrite_episode_parquet, "episodes", workers)
    verify_dataset(root)


def verify_dataset(root: Path) -> None:
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        return
    columns = [c for c in pq.read_schema(data_files[0]).names if c not in IGNORED_DATA_COLUMNS]
    info = json.loads((root / "meta" / "info.json").read_text())
    missing = sorted(set(columns) - set(info.get("features", {})))
    if missing:
        raise RuntimeError(f"{root} has parquet columns absent from info.json: {missing}")
    log.info("verified feature columns: %s", columns)


def convert_dataset(data_root: Path, repo_prefix: str, workers: int) -> None:
    splits = ["splitA", "splitB", "splitC", "splitD"]

    def run(split: str) -> int:
        root = data_root / split
        if not (root / "meta" / "info.json").exists():
            log.warning("skip missing split: %s", root)
            return 0
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.convert_dataset_v21_to_v30",
            f"--repo-id={repo_prefix}/{split}",
            f"--root={root}",
            "--push-to-hub=false",
            "--force-conversion",
        ]
        return subprocess.run(cmd).returncode

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, split): split for split in splits}
        for future in as_completed(futures):
            split = futures[future]
            code = future.result()
            if code != 0:
                raise RuntimeError(f"{split} conversion failed with code {code}")


def merge_abc(data_root: Path, output_dir: Path, repo_prefix: str) -> None:
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (output_dir / "meta" / "info.json").exists():
        log.info("merged dataset exists: %s", output_dir)
        return
    datasets = []
    for split in ["splitA", "splitB", "splitC"]:
        root = data_root / split
        dataset = LeRobotDataset(f"{repo_prefix}/{split}", root=str(root))
        log.info("%s episodes=%d frames=%d", split, dataset.num_episodes, dataset.num_frames)
        datasets.append(dataset)
    merged = merge_datasets(datasets, output_repo_id=f"{repo_prefix}/mergedABC", output_dir=str(output_dir))
    log.info("merged episodes=%d frames=%d path=%s", merged.num_episodes, merged.num_frames, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["convert", "normalize", "merge", "all"])
    parser.add_argument("--data-root", default="data/calvin-lerobot")
    parser.add_argument("--merged-root", default="data/merged_ABC")
    parser.add_argument("--repo-prefix", default="calvin")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    merged_root = Path(args.merged_root).expanduser().resolve()

    if args.stage in {"convert", "all"}:
        convert_dataset(data_root, args.repo_prefix, args.workers)
    if args.stage in {"normalize", "all"}:
        for split in ["splitA", "splitB", "splitC", "splitD"]:
            normalize_dataset(data_root / split, args.workers)
    if args.stage in {"merge", "all"}:
        merge_abc(data_root, merged_root, args.repo_prefix)
        normalize_dataset(merged_root, args.workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
