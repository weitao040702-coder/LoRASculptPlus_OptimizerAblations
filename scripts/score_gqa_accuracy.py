#!/usr/bin/env python3
"""Score GQA testdev-balanced accuracy without optional scene graphs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    predictions = {str(x["questionId"]): str(x["prediction"]).strip().lower() for x in json.loads(args.predictions.read_text())}
    balanced = [q for q in questions.values() if q.get("isBalanced")]
    missing = [qid for qid, q in questions.items() if q.get("isBalanced") and qid not in predictions]
    if missing:
        raise RuntimeError(f"missing predictions: {len(missing)}")
    correct = sum(predictions[qid] == str(q["answer"]).strip().lower() for qid, q in questions.items() if q.get("isBalanced"))
    accuracy = 100.0 * correct / len(balanced)
    text = f"Samples: {len(balanced)}\nAccuracy: {accuracy:.2f}%\n"
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
