import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def count_lfs_pointers(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.parquet") if path.stat().st_size < 512)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="xiaoma26/calvin-lerobot")
    parser.add_argument("--output-dir", default="data/calvin-lerobot")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    proxies = None
    if args.proxy:
        proxies = {"http": args.proxy, "https": args.proxy}
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    before = count_lfs_pointers(output_dir)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        revision=args.revision,
        ignore_patterns=[".git", ".gitattributes"],
        proxies=proxies,
    )
    after = count_lfs_pointers(output_dir)
    print(f"dataset: {args.repo_id}")
    print(f"saved_to: {output_dir}")
    print(f"lfs_pointers_before: {before}")
    print(f"lfs_pointers_after: {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
