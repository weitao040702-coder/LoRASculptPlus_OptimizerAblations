#!/usr/bin/env python3
"""Mark one experiment as failed without hiding its log path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llava.train.progress import update_run_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    state = json.loads(args.progress.read_text(encoding="utf-8")) if args.progress.is_file() else {}
    state.update(
        {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "FAILED",
            "failure_exit_code": args.exit_code,
            "failure_log": str(args.log),
            "log": str(args.log),
        }
    )
    update_run_progress(args.progress, state)


if __name__ == "__main__":
    main()
