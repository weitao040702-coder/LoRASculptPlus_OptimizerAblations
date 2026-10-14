#!/usr/bin/env python3
"""Build the LoRASculpt+ rank-16/32/64 paper-style comparison table."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAPER = {
    16: {"OKVQA": 57.34, "OCRVQA": 62.10, "GQA": 60.96, "TextVQA": 57.40, "Source": 59.45, "Target": 85.21, "Avg": 72.33},
    32: {"OKVQA": 57.23, "OCRVQA": 64.50, "GQA": 60.86, "TextVQA": 57.38, "Source": 59.99, "Target": 85.83, "Avg": 72.91},
    64: {"OKVQA": 56.65, "OCRVQA": 63.90, "GQA": 59.84, "TextVQA": 55.48, "Source": 58.97, "Target": 85.28, "Avg": 72.12},
}

DATASET_FILES = {
    "OKVQA": "okvqa/llava-v1.5-7b/result-okvqa.txt",
    "OCRVQA": "ocrvqa/llava-v1.5-7b/result-ocrvqa.txt",
    "GQA": "gqa/llava-v1.5-7b/result-gqa.txt",
    "TextVQA": "textvqa/llava-v1.5-7b/result-textvqa.txt",
    "Target": "iconqa/llava-v1.5-7b/result-iconqa.txt",
}


def read_score(path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Accuracy(?: on [^:]+)?:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    return float(match.group(1)) if match else None


def fmt(value: float | None) -> str:
    return "pending" if value is None else f"{value:.2f}"


def local_row(root: Path, rank: int) -> dict[str, float | None]:
    checkpoint = root / (
        "checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculptPlus-10k-"
        f"lora-r{rank}-a64-connector-r128-ca64-e3-srr0.1-cmr1e-3-1e-5-ortho1e-5"
    )
    result_root = checkpoint / "eval_results"
    values = {name: read_score(result_root / relpath) for name, relpath in DATASET_FILES.items()}
    upstream = [values[name] for name in ("OKVQA", "OCRVQA", "GQA", "TextVQA")]
    values["Source"] = sum(upstream) / 4 if all(value is not None for value in upstream) else None
    values["Avg"] = (
        (values["Source"] + values["Target"]) / 2
        if values["Source"] is not None and values["Target"] is not None
        else None
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "rank16_32_64_paper_style_results.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# LoRASculpt+ Rank Comparison (LLaVA-v1.5-7B)",
        "",
        "| Methods | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source | Target (IconQA) | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for rank in (16, 32, 64):
        official = PAPER[rank]
        local = local_row(args.root, rank)
        lines.append(
            f"| LoRASculpt+ (paper) | {rank} | {fmt(official['OKVQA'])} | {fmt(official['OCRVQA'])} | "
            f"{fmt(official['GQA'])} | {fmt(official['TextVQA'])} | {fmt(official['Source'])} | "
            f"{fmt(official['Target'])} | {fmt(official['Avg'])} |"
        )
        lines.append(
            f"| LoRASculpt+ (local reproduction) | {rank} | {fmt(local['OKVQA'])} | {fmt(local['OCRVQA'])} | "
            f"{fmt(local['GQA'])} | {fmt(local['TextVQA'])} | {fmt(local['Source'])} | "
            f"{fmt(local['Target'])} | {fmt(local['Avg'])} |"
        )
        for method, values in (("paper", official), ("local", local)):
            csv_rows.append({"method": method, "rank": rank, **values})

    lines.extend([
        "",
        "> Source is the mean of OKVQA, OCRVQA, GQA, and TextVQA. Target is IconQA and Avg is (Source + Target) / 2, matching Eq. (19) and Table 1. The paper rows are the official LLaVA-v1.5-7B references. Local OCRVQA uses the deterministic 5,000-example subset and local GQA falls back to testdev-balanced accuracy when the scene-graph evaluator is unavailable; local rows must be interpreted as local reproduction results.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "rank", "OKVQA", "OCRVQA", "GQA", "TextVQA", "Source", "Target", "Avg"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output)
    print(csv_output)


if __name__ == "__main__":
    main()
