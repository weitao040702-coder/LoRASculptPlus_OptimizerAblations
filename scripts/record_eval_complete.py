#!/usr/bin/env python3
"""Record evaluation timing, prediction counts, and peak VRAM."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from llava.train.progress import update_run_progress


EXPECTED = {"OKVQA": 5046, "OCRVQA": 5000, "GQA": 12578, "TextVQA": 5000}
PREDICTIONS = {
    "OKVQA": "okvqa/llava-v1.5-7b/llava-v1.5-7b-okvqa.jsonl",
    "OCRVQA": "ocrvqa/llava-v1.5-7b/llava-v1.5-7b-ocrvqa.jsonl",
    "GQA": "gqa/llava-v1.5-7b/llava-v1.5-7b-gqa.jsonl",
    "TextVQA": "textvqa/llava-v1.5-7b/llava-v1.5-7b-textvqa.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    state = json.loads(args.progress.read_text(encoding="utf-8"))
    peak = 0
    model = "unknown"
    if args.gpu_log.is_file():
        with args.gpu_log.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                try:
                    peak = max(peak, int(float(row["memory.used [MiB]"])))
                except (KeyError, TypeError, ValueError):
                    pass
                if row.get("name"):
                    model = row["name"]
    counts = {}
    for dataset, relative in PREDICTIONS.items():
        path = args.result_root / relative
        counts[dataset] = sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.is_file() else 0
    state.update(
        {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "COMPLETED",
            "eval_time_sec": max(0, args.end - args.start),
            "eval_start_epoch": args.start,
            "eval_end_epoch": args.end,
            "eval_samples": counts,
            "expected_eval_samples": EXPECTED,
            "eval_peak_vram_mib": peak,
            "peak_vram_mib": max(int(state.get("peak_vram_mib", 0)), peak),
            "gpu_model": model if model != "unknown" else state.get("gpu_model", "unknown"),
            "evaluation_protocol": {
                "temperature": 0,
                "OCRVQA": "deterministic 5000-example subset",
                "GQA": "12578 testdev-balanced accuracy fallback when scene-graph evaluator is unavailable",
            },
        }
    )
    update_run_progress(args.progress, state)


if __name__ == "__main__":
    main()
