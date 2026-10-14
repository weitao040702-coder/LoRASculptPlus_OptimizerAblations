#!/usr/bin/env python3
"""Fail closed unless all six runs have checkpoints and four evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = {"ABManifold": "AB-manifold", "Muon": "Muon"}
DATASETS = {
    "OKVQA": "okvqa/llava-v1.5-7b/result-okvqa.txt",
    "OCRVQA": "ocrvqa/llava-v1.5-7b/result-ocrvqa.txt",
    "GQA": "gqa/llava-v1.5-7b/result-gqa.txt",
    "TextVQA": "textvqa/llava-v1.5-7b/result-textvqa.txt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    errors = []
    for method in METHODS:
        for rank in (16, 32, 64):
            checkpoint = args.root / "checkpoints" / f"{method}-r{rank}"
            if not (checkpoint / "adapter_model.bin").is_file():
                errors.append(f"missing adapter_model.bin: {checkpoint}")
            if not (checkpoint / "adapter_config.json").is_file():
                errors.append(f"missing adapter_config.json: {checkpoint}")
            if not (checkpoint / "config.json").is_file():
                errors.append(f"missing config.json: {checkpoint}")
            prediction_dir = checkpoint / "eval_results"
            for dataset, relative in DATASETS.items():
                result = prediction_dir / relative
                if not result.is_file() or result.stat().st_size == 0:
                    errors.append(f"missing {dataset} result: {result}")
                prediction_name = {
                    "OKVQA": "llava-v1.5-7b-okvqa.jsonl",
                    "OCRVQA": "llava-v1.5-7b-ocrvqa.jsonl",
                    "GQA": "llava-v1.5-7b-gqa.jsonl",
                    "TextVQA": "llava-v1.5-7b-textvqa.jsonl",
                }[dataset]
                prediction = result.parent / prediction_name
                if not prediction.is_file() or prediction.stat().st_size == 0:
                    errors.append(f"missing {dataset} predictions: {prediction}")
    for required in ("results.md", "results.csv", "run_manifest.json", "progress/progress.md"):
        path = args.root / required
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing final artifact: {path}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("VALIDATION=PASS all six checkpoints and four-dataset evaluations present")


if __name__ == "__main__":
    main()
