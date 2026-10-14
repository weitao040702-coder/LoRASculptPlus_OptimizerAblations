#!/usr/bin/env python3
"""Convert official IconQA choose-text data to LLaVA JSON/JSONL format."""

import argparse
import json
import random
from pathlib import Path


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_records(root, split_map, problems, split_name, check_images=True):
    records = []
    for pid in split_map[f"choose_txt_{split_name}"]:
        item = problems[str(pid)]
        choices = item["choices"]
        answer = str(choices[int(item["answer"])])
        rel_image = Path("iconqa") / split_name / "choose_txt" / str(pid) / "image.png"
        image_path = root / rel_image
        if check_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        question = item["question"]
        prompt = question + "\nChoices: " + " | ".join(choices)
        records.append({
            "question_id": str(pid),
            "image": rel_image.as_posix(),
            "conversations": [
                {"from": "human", "value": "<image>\n" + prompt},
                {"from": "gpt", "value": answer},
            ],
            "answer": answer,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problems = load_json(args.data_root / "problems.json")
    split_map = load_json(args.data_root / "pid_splits.json")
    args.output_root.mkdir(parents=True, exist_ok=True)

    train = build_records(args.data_root, split_map, problems, "train", check_images=False)
    if args.train_sample_size is not None and args.train_sample_size < len(train):
        train = random.Random(args.seed).sample(train, args.train_sample_size)
    test = build_records(args.data_root, split_map, problems, "test", check_images=True)

    train_name = "iconqa_txt-train.json" if args.train_sample_size is None else f"iconqa_txt-train-{args.train_sample_size}k-seed{args.seed}.json"
    with (args.output_root / train_name).open("w", encoding="utf-8") as handle:
        json.dump(train, handle, ensure_ascii=False, indent=2)
    with (args.output_root / "fixed_iconqa_txt-test.jsonl").open("w", encoding="utf-8") as handle:
        for record in test:
            prompt = record["conversations"][0]["value"].replace("<image>\n", "", 1)
            json.dump({
                "question_id": record["question_id"],
                "image": record["image"],
                "text": prompt,
                "answer": record["answer"],
            }, handle, ensure_ascii=False)
            handle.write("\n")

    print(f"train={len(train)} test={len(test)} output={args.output_root}")


if __name__ == "__main__":
    main()
