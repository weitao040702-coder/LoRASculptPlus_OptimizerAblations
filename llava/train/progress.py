"""Atomic per-run and aggregate progress files for optimizer ablations."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from pathlib import Path


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _state_files(progress_dir: Path, current: Path):
    for path in sorted(progress_dir.glob("*.json")):
        if path.name in {"progress.json", current.name}:
            continue
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    if current.exists():
        try:
            yield json.loads(current.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass


def render_progress(progress_dir: Path, allocation_job_id: str = "", node: str = "") -> dict:
    states = list(_state_files(progress_dir, progress_dir / "__current__.json"))
    order = {"ABManifold": 0, "Muon": 1, "ab_manifold": 0, "muon": 1}
    states.sort(key=lambda item: (order.get(str(item.get("method")), 9), int(item.get("rank", 0))))
    aggregate = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "allocation_job_id": allocation_job_id or os.environ.get("OPTAB_ALLOCATION_JOB", ""),
        "node": node or os.environ.get("OPTAB_NODE", ""),
        "gpus": int(os.environ.get("OPTAB_GPU_COUNT", "3")),
        "experiments": states,
    }
    _atomic_write(progress_dir / "progress.json", json.dumps(aggregate, indent=2, sort_keys=True) + "\n")

    columns = [
        "Method", "Rank", "GPU", "Smoke", "Train", "Checkpoint",
        "OKVQA", "OCRVQA", "GQA", "TextVQA", "Status", "Step", "Last Update",
    ]
    lines = [
        "# Optimizer Ablation Progress",
        "",
        f"Updated: {aggregate['updated_at']}    Allocation: {aggregate['allocation_job_id'] or 'unknown'}    Node: {aggregate['node'] or 'unknown'}",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    tsv = ["\t".join(columns)]
    for item in states:
        row = [
            item.get("method", ""), item.get("rank", ""), item.get("gpu", ""),
            item.get("smoke", "pending"), item.get("train", item.get("status", "pending")),
            item.get("checkpoint_status", "pending"), item.get("OKVQA", "pending"),
            item.get("OCRVQA", "pending"), item.get("GQA", "pending"),
            item.get("TextVQA", "pending"), item.get("status", "WAITING"),
            f"{item.get('current_step', 0)}/{item.get('total_steps', 0)}",
            item.get("updated_at", ""),
        ]
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
        tsv.append("\t".join(str(value) for value in row))
    _atomic_write(progress_dir / "progress.md", "\n".join(lines) + "\n")
    _atomic_write(progress_dir / "progress.tsv", "\n".join(tsv) + "\n")
    return aggregate


def update_run_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lock_path = path.parent / ".progress.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        render_progress(path.parent)
        fcntl.flock(lock, fcntl.LOCK_UN)
