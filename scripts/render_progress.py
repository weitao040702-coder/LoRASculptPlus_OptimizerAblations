#!/usr/bin/env python
"""Render progress.md/progress.tsv/progress.json from per-run state files."""

import argparse
from pathlib import Path

from llava.train.progress import render_progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-dir", type=Path, default=Path("progress"))
    parser.add_argument("--allocation-job", default="")
    parser.add_argument("--node", default="")
    args = parser.parse_args()
    render_progress(args.progress_dir, args.allocation_job, args.node)


if __name__ == "__main__":
    main()
