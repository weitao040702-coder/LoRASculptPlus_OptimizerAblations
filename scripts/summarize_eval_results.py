#!/usr/bin/env python3
"""Create a compact paper-style table from LoRASculpt evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DATASETS = ["IconQA", "GQA", "TextVQA", "OK-VQA", "OCR-VQA"]


def read_score(path: Path):
    if not path.is_file():
        return None, "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Accuracy(?: on [^:]+)?:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if match:
        source = "GQA testdev-balanced accuracy scorer" if path.name == "result-gqa.txt" else "official evaluator"
        return float(match.group(1)), source
    match = re.search(r"overall['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1)), "official evaluator"
    return None, "unparsed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    root = args.checkpoint
    results_root = args.results_dir or root / "eval_results"
    paths = {
        "IconQA": results_root / "iconqa/llava-v1.5-7b/result-iconqa.txt",
        "GQA": results_root / "gqa/llava-v1.5-7b/result-gqa.txt",
        "TextVQA": results_root / "textvqa/llava-v1.5-7b/result-textvqa.txt",
        "OK-VQA": results_root / "okvqa/llava-v1.5-7b/result-okvqa.txt",
        "OCR-VQA": results_root / "ocrvqa/llava-v1.5-7b/result-ocrvqa.txt",
    }
    rows = []
    for dataset in DATASETS:
        score, source = read_score(paths[dataset])
        rows.append({"Dataset": dataset, "Accuracy (%)": "" if score is None else f"{score:.2f}", "Source": source})

    output = args.output or root / "paper_style_results.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = {r["Dataset"]: float(r["Accuracy (%)"]) for r in rows if r["Accuracy (%)"]}
    rank = args.rank
    if rank is None:
        match = re.search(r"lora-r(\d+)", root.name)
        rank = int(match.group(1)) if match else 0
    upstream_names = ["OK-VQA", "OCR-VQA", "GQA", "TextVQA"]
    upstream_scores = [scores[name] for name in upstream_names if name in scores]
    upstream_avg = sum(upstream_scores) / len(upstream_scores) if len(upstream_scores) == len(upstream_names) else None
    source_avg = f"{upstream_avg:.2f}" if upstream_avg is not None else "pending"
    paper_table = [
        "# LoRASculpt Upstream Reproduction Results",
        "",
        "| Methods | OKVQA | OCRVQA | GQA | TextVQA | Source Avg | Target Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| LoRA Rank-{rank} | {scores.get('OK-VQA', 'pending')} | {scores.get('OCR-VQA', 'pending')} | {scores.get('GQA', 'pending')} | {scores.get('TextVQA', 'pending')} | {source_avg} | N/A |",
        "",
        "> Source Avg is the mean of the four upstream datasets. Target Avg is N/A because this run evaluated upstream data only; IconQA target evaluation was not included. GQA is testdev-balanced accuracy only; optional scene-graph-dependent metrics were not computed.",
        "",
    ]
    output.write_text("\n".join(paper_table), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Dataset", "Accuracy (%)", "Source"])
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    print(output.with_suffix(".csv"))


if __name__ == "__main__":
    main()
