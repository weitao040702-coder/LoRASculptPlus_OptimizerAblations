#!/usr/bin/env python
"""Record one upstream result in a run's progress state."""

import argparse
import json
import re
import time
from pathlib import Path

from llava.train.progress import update_run_progress


def read_score(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Accuracy(?: on [^:]+)?:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if match:
        return float(match.group(1))
    match = re.search(r"['\"]overall['\"]\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--status", default="EVALUATING")
    args = parser.parse_args()
    path = args.root / "progress" / f"{args.method}-r{args.rank}.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state.update({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": args.method, "rank": args.rank, "gpu": args.gpu, "status": args.status})
    if args.dataset and args.result:
        value = read_score(args.result)
        state[args.dataset] = "pending" if value is None else f"{value:.2f}"
    update_run_progress(path, state)


if __name__ == "__main__":
    main()
