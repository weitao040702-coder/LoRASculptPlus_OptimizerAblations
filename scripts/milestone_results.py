#!/usr/bin/env python3
"""Print the recorded four-benchmark means for the milestone comparison."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'milestone_results'
FILES = ('original-paper-seed42.json', 'orbit-paper-seed42.json')


def main():
    """Read stored experiment summaries without changing measured scores."""
    for filename in FILES:
        for run in json.loads((ROOT / filename).read_text()):
            scores = run['scores']
            mean = run['source_mean']
            print(f"{filename.removesuffix('.json'):35} rank={run['rank']:2} mean={mean:.2f}")


if __name__ == '__main__':
    main()
