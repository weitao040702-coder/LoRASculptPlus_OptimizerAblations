#!/usr/bin/env python
"""Build results.md/results.csv from completed upstream evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


LOCAL = {
    16: {"OKVQA": 57.40, "OCRVQA": 63.42, "GQA": 61.22, "TextVQA": 57.79},
    32: {"OKVQA": 57.30, "OCRVQA": 62.98, "GQA": 61.38, "TextVQA": 58.05},
    64: {"OKVQA": 57.46, "OCRVQA": 62.74, "GQA": 61.81, "TextVQA": 57.82},
}
PAPER = {
    16: {"OKVQA": 57.34, "OCRVQA": 62.10, "GQA": 60.96, "TextVQA": 57.40},
    32: {"OKVQA": 57.23, "OCRVQA": 64.50, "GQA": 60.86, "TextVQA": 57.38},
    64: {"OKVQA": 56.65, "OCRVQA": 63.90, "GQA": 59.84, "TextVQA": 55.48},
}
FILES = {
    "OKVQA": "okvqa/llava-v1.5-7b/result-okvqa.txt",
    "OCRVQA": "ocrvqa/llava-v1.5-7b/result-ocrvqa.txt",
    "GQA": "gqa/llava-v1.5-7b/result-gqa.txt",
    "TextVQA": "textvqa/llava-v1.5-7b/result-textvqa.txt",
}


def score(path: Path):
    if not path.is_file():
        return None
    match = re.search(r"Accuracy(?: on [^:]+)?:\s*([0-9]+(?:\.[0-9]+)?)%", path.read_text(errors="replace"))
    if match:
        return float(match.group(1))
    match = re.search(r"['\"]overall['\"]\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", path.read_text(errors="replace"))
    return float(match.group(1)) if match else None


def fmt(value):
    return "pending" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    rows = []
    root = args.root
    for method, directory in (("AB-manifold", "ABManifold"), ("Muon", "Muon")):
        for rank in (16, 32, 64):
            checkpoint = root / "checkpoints" / f"{directory}-r{rank}"
            values = {name: score(checkpoint / "eval_results" / relative) for name, relative in FILES.items()}
            complete = all(value is not None for value in values.values())
            source = sum(values.values()) / 4 if complete else None
            local_source = sum(LOCAL[rank].values()) / 4
            paper_source = sum(PAPER[rank].values()) / 4
            rows.append({
                "Method": method,
                "Rank": rank,
                **values,
                "Source Avg": source,
                "Delta vs Local": None if source is None else source - local_source,
                "Delta vs Paper": None if source is None else source - paper_source,
                "Status": "COMPLETED" if complete else "INCOMPLETE",
                "Checkpoint": str(checkpoint),
            })

    fields = ["Method", "Rank", "OKVQA", "OCRVQA", "GQA", "TextVQA", "Source Avg", "Delta vs Local", "Delta vs Paper", "Status", "Checkpoint"]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})

    lines = [
        "# Optimizer Ablation Results",
        "",
        "The source average is the mean of OKVQA, OCRVQA, GQA, and TextVQA.",
        "",
        "| Method | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source Avg | Delta vs Local LoRASculpt+ | Delta vs Paper | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append("| {Method} | {Rank} | {OKVQA} | {OCRVQA} | {GQA} | {TextVQA} | {Source Avg} | {Delta vs Local} | {Delta vs Paper} | {Status} |".format(
            **{key: fmt(row[key]) if key not in {"Method", "Rank", "Status", "Checkpoint"} else row[key] for key in row}
        ))
    lines += [
        "",
        "## Protocol",
        "",
        "IconQA was used for training data only. Evaluation contains OKVQA, OCRVQA, GQA, and TextVQA.",
        "OCRVQA uses the deterministic local 5,000-example subset. GQA uses testdev-balanced accuracy when the scene-graph evaluator is unavailable.",
        "The original orthogonal loss is disabled with ORTHO_LAMBDA=0. SRR and both CMR terms remain enabled.",
        "",
    ]
    (root / "results.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"rows": rows, "local_baseline": LOCAL, "paper_reference": PAPER, "orthogonal_loss": False}
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
