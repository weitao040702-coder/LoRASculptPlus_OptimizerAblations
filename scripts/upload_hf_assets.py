"""Resumable HF upload of assets and checkpoints; never upload environments."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="weitao040702/LoRASculptPlus_OptimizerAblations")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    api = HfApi()
    # Existing repository only: never alter visibility while resuming uploads.
    info = api.repo_info(args.repo, repo_type="dataset")
    print("HF repository:", info.id, "private:", info.private, flush=True)
    stage = ROOT / ".hf-upload-stage"
    stage.mkdir(exist_ok=True)
    records = []
    # Per-file links let pathlib discover files under assets' directory link
    # without copying 160+ GB. Links are followed when uploading file bytes.
    for relative in ("assets/data", "assets/models", "checkpoints"):
        source = (ROOT / relative).resolve(strict=True)
        for directory, dirs, files in os.walk(source, followlinks=True):
            dirs[:] = [d for d in dirs if d not in {".git", ".cache", "__pycache__"}]
            for name in files:
                original = Path(directory) / name
                path = Path(relative) / original.relative_to(source)
                destination = stage / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_symlink() and destination.exists():
                    raise RuntimeError(f"Unexpected existing stage file: {destination}")
                if destination.is_symlink():
                    if destination.resolve() != original.resolve():
                        raise RuntimeError(f"Stage link differs: {destination}")
                else:
                    destination.symlink_to(original.resolve(strict=True))
                records.append({"path": path.as_posix(), "size": original.stat().st_size})
    manifest = {"repo": args.repo, "created_at": datetime.now(timezone.utc).isoformat(),
                "files": records, "total_bytes": sum(r["size"] for r in records)}
    (stage / "asset_inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Files:", len(records), "Bytes:", manifest["total_bytes"], flush=True)
    api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=stage,
                            num_workers=args.workers, print_report_every=60)
    remote = {f.path: f for f in api.list_repo_tree(args.repo, repo_type="dataset", recursive=True)
              if hasattr(f, "size")}
    errors = [r["path"] for r in records
              if r["path"] not in remote or remote[r["path"]].size != r["size"]]
    if errors:
        raise RuntimeError(f"Remote verification failed for {len(errors)} files: {errors[:10]}")
    print("VERIFIED: all planned file paths and sizes match HF repository", flush=True)


if __name__ == "__main__":
    main()
