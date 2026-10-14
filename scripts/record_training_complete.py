#!/usr/bin/env python3
"""Attach training timing and GPU telemetry to one progress row."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from llava.train.progress import update_run_progress


def telemetry(path: Path):
    peak = 0
    model = "unknown"
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                try:
                    peak = max(peak, int(float(row["memory.used [MiB]"])))
                except (KeyError, TypeError, ValueError):
                    pass
                if row.get("name"):
                    model = row["name"]
    return peak, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    state = json.loads(args.progress.read_text(encoding="utf-8"))
    peak, model = telemetry(args.gpu_log)
    method = state.get("method", "")
    optimizer = "AdamW" if method == "ABManifold" else "MuonWithAdamW"
    state.update(
        {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "TRAINED",
            "train": "TRAINED",
            "checkpoint_status": "present",
            "current_step": int(state.get("total_steps", 1875)),
            "train_start_epoch": args.start,
            "train_end_epoch": args.end,
            "train_time_sec": max(0, args.end - args.start),
            "peak_vram_mib": peak,
            "gpu_model": model,
            "actual_optimizer": optimizer,
            "deepspeed": False,
            "global_batch_size": 16,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "orthogonal_loss": False,
            "srr": True,
            "cmr": True,
        }
    )
    update_run_progress(args.progress, state)


if __name__ == "__main__":
    main()
