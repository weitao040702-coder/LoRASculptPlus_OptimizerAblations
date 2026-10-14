#!/usr/bin/env python3
"""Verify files required by the four LoRASculpt upstream evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_jsonl(path: Path, image_root: Path) -> int:
    rows = load_lines(path)
    missing = [row["image"] for row in rows if not (image_root / row["image"]).is_file()]
    if missing:
        raise RuntimeError(f"{path}: missing {len(missing)} images, first={missing[0]}")
    print(f"OK {path}: rows={len(rows)} images={len({row['image'] for row in rows})}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.eval_root
    check_jsonl(root / "gqa/llava_gqa_testdev_balanced.jsonl", root / "gqa/data/images")
    check_jsonl(root / "textvqa/llava_textvqa_val_v051_ocr.jsonl", root / "textvqa/train_images")
    check_jsonl(root / "okvqa/okvqa_val.jsonl", root / "okvqa/val2014")
    check_jsonl(root / "ocrvqa/sampled_ocrvqa_test.jsonl", root / "ocrvqa/sampled_images")
    for path in [
        root / "okvqa/mscoco_val2014_annotations.json",
        root / "okvqa/OpenEnded_mscoco_val2014_questions.json",
        root / "textvqa/TextVQA_0.5.1_val.json",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"OK {path}")


if __name__ == "__main__":
    main()
