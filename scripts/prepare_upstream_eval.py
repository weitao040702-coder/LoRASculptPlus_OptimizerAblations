#!/usr/bin/env python3
"""Prepare the four upstream evaluation datasets in LoRASculpt format."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def load_ocrvqa_parquet(download_root: Path):
    """Load OCR-VQA validation metadata without materializing image bytes."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("OCR-VQA Parquet input requires pyarrow") from exc

    records = []
    paths = sorted(download_root.glob("ocrvqa-p*.parquet"))
    if not paths:
        return records
    for path in paths:
        table = parquet.read_table(path, columns=["image_id", "questions", "answers"])
        values = table.to_pydict()
        for image_id, questions, answers in zip(
            values["image_id"], values["questions"], values["answers"]
        ):
            records.append({
                "image_id": str(image_id),
                "questions": questions or [],
                "answers": answers or [],
            })
    return records


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_okvqa(root: Path) -> None:
    dataset_root = root / "okvqa"
    question_path = dataset_root / "OpenEnded_mscoco_val2014_questions.json"
    if not question_path.is_file():
        raise FileNotFoundError(question_path)
    questions = load_json(question_path)["questions"]
    rows = []
    for item in questions:
        image_id = int(item["image_id"])
        rows.append({
            "question_id": str(item["question_id"]),
            "image": f"COCO_val2014_{image_id:012d}.jpg",
            "text": item["question"] + "\nAnswer the question using a single word or phrase.",
            "category": "default",
        })
    write_jsonl(dataset_root / "okvqa_val.jsonl", rows)
    print(f"OK-VQA questions: {len(rows)}")


def prepare_ocrvqa(root: Path, sample_size: int = 5000) -> None:
    """Convert common OCR-VQA archive layouts to the LoRASculpt JSONL format.

    The public OCR-VQA archive has appeared in several layouts. We search JSON
    files for records containing an image id and question/answer fields, then
    choose a deterministic test/validation-like subset and locate matching
    image files in the extracted archive.
    """

    raw_root = root / "ocrvqa" / "raw"
    json_files = sorted(raw_root.rglob("*.json"))
    parquet_records = []
    if not json_files:
        parquet_records = load_ocrvqa_parquet(root.parent / "upstream" / "downloads")
        if not parquet_records:
            raise FileNotFoundError(f"No OCR-VQA JSON or Parquet metadata found")

    candidates = []
    for path in json_files:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            values = data.get("data") if isinstance(data.get("data"), list) else None
            if values is None:
                values = data.get("questions") if isinstance(data.get("questions"), list) else None
            if values is None:
                values = [data]
        elif isinstance(data, list):
            values = data
        else:
            continue
        score = sum(1 for item in values[:50] if isinstance(item, dict) and (
            any(k in item for k in ("image_id", "imageID", "image", "image_path"))
            and any(k in item for k in ("question", "questions"))
        ))
        if score:
            candidates.append((score, path, values))

    if candidates:
        _, metadata_path, records = max(candidates, key=lambda item: (item[0], str(item[1])))
        print(f"OCR-VQA metadata: {metadata_path} records={len(records)}")
    else:
        records = parquet_records
        print(f"OCR-VQA metadata: validation Parquet records={len(records)}")

    image_root = raw_root / "images"
    image_files = {}
    if not parquet_records:
        for image_path in raw_root.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_files.setdefault(image_path.stem, image_path)
                image_files.setdefault(image_path.name, image_path)

    rows = []
    for item in records:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id", item.get("imageID", item.get("image")))
        questions = item.get("questions", item.get("question"))
        answers = item.get("answers", item.get("answer"))
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]
        if not image_id or not questions or not answers:
            continue
        image_id = str(image_id)
        image_name = Path(image_id).name
        if Path(image_name).suffix:
            image_rel = image_name
            found = image_files.get(image_name) or image_files.get(Path(image_name).stem)
        else:
            found = image_files.get(image_name)
            image_rel = image_name + ".jpg"
            if found is None:
                for suffix in (".jpg", ".jpeg", ".png"):
                    if image_id + suffix in image_files:
                        found = image_files[image_id + suffix]
                        image_rel = image_id + suffix
                        break
        if found is None and parquet_records:
            image_rel = image_id + ".jpg"
            direct_path = raw_root / "images" / image_rel
            if direct_path.is_file():
                found = direct_path
        if found is None:
            continue
        for question_index, (question, answer) in enumerate(zip(questions, answers)):
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            if isinstance(question, dict):
                question = question.get("question", "")
            rows.append({
                "question_id": f"{image_id}_{question_index}",
                "image": image_rel,
                "text": str(question) + "\nAnswer the question using a single word or phrase.",
                "answer": str(answer),
                "category": "default",
            })

    if not rows:
        raise RuntimeError("No OCR-VQA records matched extracted images")
    random.Random(0).shuffle(rows)
    rows = rows[:sample_size]
    sampled_root = root / "ocrvqa" / "sampled_images"
    sampled_root.mkdir(parents=True, exist_ok=True)
    selected = set()
    for row in rows:
        selected.add(row["image"])
    for image_name in selected:
        source = image_files.get(image_name) or image_files.get(Path(image_name).stem)
        if source is None:
            continue
        target = sampled_root / image_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and not target.exists():
            target.unlink()
        if not target.exists():
            target.symlink_to(source.resolve())
    write_jsonl(root / "ocrvqa" / "sampled_ocrvqa_test.jsonl", rows)
    print(f"OCR-VQA samples: {len(rows)} images={len(selected)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--ocr-sample-size", type=int, default=5000)
    args = parser.parse_args()
    prepare_okvqa(args.eval_root)
    prepare_ocrvqa(args.eval_root, args.ocr_sample_size)


if __name__ == "__main__":
    main()
