#!/usr/bin/env python3
"""Build the final optimizer-ablation report from real artifacts only."""

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
DATASETS = ("OKVQA", "OCRVQA", "GQA", "TextVQA")
RESULT_FILES = {
    "OKVQA": "okvqa/llava-v1.5-7b/result-okvqa.txt",
    "OCRVQA": "ocrvqa/llava-v1.5-7b/result-ocrvqa.txt",
    "GQA": "gqa/llava-v1.5-7b/result-gqa.txt",
    "TextVQA": "textvqa/llava-v1.5-7b/result-textvqa.txt",
}


def score(path: Path):
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    match = re.search(r"Accuracy(?: on [^:]+)?:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if match:
        return float(match.group(1))
    match = re.search(r"['\"]overall['\"]\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def state_for(root: Path, directory: str):
    path = root / "progress" / f"{directory}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def fmt(value):
    return "pending" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    rows = []
    for method, directory in (("AB-manifold", "ABManifold"), ("Muon", "Muon")):
        for rank in (16, 32, 64):
            checkpoint = root / "checkpoints" / f"{directory}-r{rank}"
            state = state_for(root, f"{directory}-r{rank}")
            values = {name: score(checkpoint / "eval_results" / relative) for name, relative in RESULT_FILES.items()}
            source_avg = sum(values.values()) / 4 if all(value is not None for value in values.values()) else None
            local_avg = sum(LOCAL[rank].values()) / 4
            paper_avg = sum(PAPER[rank].values()) / 4
            status = "COMPLETED" if source_avg is not None else state.get("status", "INCOMPLETE")
            rows.append(
                {
                    "Method": method,
                    "Rank": rank,
                    **values,
                    "Source Avg": source_avg,
                    "Delta vs Local LoRASculpt+": None if source_avg is None else source_avg - local_avg,
                    "Delta vs Paper": None if source_avg is None else source_avg - paper_avg,
                    "Train Time (s)": state.get("train_time_sec"),
                    "Eval Time (s)": state.get("eval_time_sec"),
                    "Peak VRAM (MiB)": state.get("peak_vram_mib"),
                    "Status": status,
                    "Checkpoint": str(checkpoint),
                    "Optimizer": state.get("actual_optimizer", "pending"),
                    "DeepSpeed": state.get("deepspeed", False),
                    "SRR": state.get("srr", True),
                    "CMR": state.get("cmr", True),
                    "ORTHO_LAMBDA": 0,
                    "AB Stats": state.get("ab_stats", {}),
                    "Muon Config": {"momentum": 0.95, "newton_schulz_steps": 5, "nesterov": True} if method == "Muon" else {},
                    "Eval Samples": state.get("eval_samples", {}),
                }
            )

    csv_fields = [
        "Method", "Rank", *DATASETS, "Source Avg", "Delta vs Local LoRASculpt+", "Delta vs Paper",
        "Train Time (s)", "Eval Time (s)", "Peak VRAM (MiB)", "Status", "Checkpoint",
    ]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in csv_fields})

    lines = [
        "# Optimizer Ablation Results",
        "",
        "Only AB-manifold and Muon are trained in this directory. The existing LoRASculpt+ runs are read-only references.",
        "Source Avg = (OKVQA + OCRVQA + GQA + TextVQA) / 4.",
        "",
        "| Method | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source Avg | Delta vs Local LoRASculpt+ | Delta vs Paper | Train Time | Eval Time | Peak VRAM | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        train = "pending" if row["Train Time (s)"] is None else f"{row['Train Time (s)'] / 3600:.2f} h"
        evaluation = "pending" if row["Eval Time (s)"] is None else f"{row['Eval Time (s)'] / 3600:.2f} h"
        vram = "pending" if row["Peak VRAM (MiB)"] is None else f"{row['Peak VRAM (MiB)']} MiB"
        lines.append(
            f"| {row['Method']} | {row['Rank']} | {fmt(row['OKVQA'])} | {fmt(row['OCRVQA'])} | "
            f"{fmt(row['GQA'])} | {fmt(row['TextVQA'])} | {fmt(row['Source Avg'])} | "
            f"{fmt(row['Delta vs Local LoRASculpt+'])} | {fmt(row['Delta vs Paper'])} | {train} | {evaluation} | {vram} | {row['Status']} |"
        )

    for reference_name, reference in (("Local LoRASculpt+", LOCAL), ("TPAMI paper", PAPER)):
        lines += ["", f"## Benchmark Deltas vs {reference_name}", "", "| Method | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source Avg |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            rank = row["Rank"]
            deltas = [None if row[name] is None else row[name] - reference[rank][name] for name in DATASETS]
            avg = None if row["Source Avg"] is None else row["Source Avg"] - sum(reference[rank].values()) / 4
            lines.append(f"| {row['Method']} | {rank} | " + " | ".join(fmt(value) for value in [*deltas, avg]) + " |")

    lines += [
        "",
        "## Configuration",
        "",
        "- Fixed training data: `assets/data/iconqa/iconqa_txt-train-10000k-seed0.json`; IconQA is not evaluated.",
        "- Backbone: LLaVA-v1.5-7B; epochs=3; global batch size=16; per-device batch=1; gradient accumulation=16.",
        "- LLM rank=16/32/64; connector rank=128; LLM and connector alpha=64.",
        "- Learning rates: LLM `2e-4`, connector `2e-5`; weight decay=0; cosine scheduler; warmup ratio=0.03.",
        "- bf16=True; tf32=True; gradient checkpointing=True; max length=2048; seed=42; DeepSpeed=False.",
        "- Original orthogonal loss is disabled with `ORTHO_LAMBDA=0`; SRR and both CMR terms remain enabled.",
        "- AB-manifold uses AdamW plus post-step masked AB-manifold retraction. Muon uses `MuonWithAdamW` with momentum=0.95, Newton-Schulz steps=5, nesterov=True.",
        "",
        "## Evaluation Protocol",
        "",
        "- Only OKVQA (5046), deterministic OCRVQA (5000), GQA testdev-balanced (12578), and TextVQA (5000) are evaluated; temperature=0.",
        "- GQA uses the local accuracy fallback if the optional scene-graph evaluator is unavailable.",
        "- Differences from the official paper protocol are limited to the deterministic local OCRVQA subset and the documented GQA fallback.",
        "",
    ]
    (root / "results.md").write_text("\n".join(lines), encoding="utf-8")

    aggregate_path = root / "progress" / "progress.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.is_file() else {}
    manifest = {
        "allocation_job_id": aggregate.get("allocation_job_id"),
        "node": aggregate.get("node"),
        "methods": ["AB-manifold", "Muon"],
        "ranks": [16, 32, 64],
        "orthogonal_loss": False,
        "config": {
            "seed": 42,
            "global_batch_size": 16,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "epochs": 3,
            "llm_learning_rate": 2e-4,
            "connector_learning_rate": 2e-5,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "deepspeed": False,
            "srr": {"warmup_steps": 100, "preserve_ratio": 0.1},
            "cmr": {"llm_lambda": 1e-3, "connector_lambda": 1e-5},
            "muon": {"momentum": 0.95, "newton_schulz_steps": 5, "nesterov": True},
        },
        "local_baseline": LOCAL,
        "paper_reference": PAPER,
        "rows": rows,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
