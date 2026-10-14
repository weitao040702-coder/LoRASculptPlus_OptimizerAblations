#!/usr/bin/env python
"""Initialize the six optimizer-ablation progress rows."""

import argparse
import json
import time
from pathlib import Path

from llava.train.progress import render_progress, update_run_progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allocation-job", default="")
    parser.add_argument("--node", default="")
    args = parser.parse_args()
    progress_dir = args.root / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for method, directory in (("ABManifold", "ABManifold"), ("Muon", "Muon")):
        for rank in (16, 32, 64):
            path = progress_dir / f"{directory}-r{rank}.json"
            if path.exists():
                continue
            update_run_progress(path, {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "method": method,
                "rank": rank,
                "gpu": "pending",
                "status": "WAITING",
                "current_step": 0,
                "total_steps": 1875,
                "smoke": "pending",
                "train": "WAITING",
                "checkpoint_status": "pending",
                "orthogonal_loss": False,
                "cmr": True,
                "srr": True,
                "checkpoint": str(args.root / "checkpoints" / f"{directory}-r{rank}"),
                "log": str(args.root / "logs" / f"{directory}-r{rank}.out"),
            })
    render_progress(progress_dir, args.allocation_job, args.node)


if __name__ == "__main__":
    main()
