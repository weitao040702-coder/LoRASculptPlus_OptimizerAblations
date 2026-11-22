#!/usr/bin/env python3
"""Summarize four scored benchmark files produced by the evaluation scripts."""
import argparse
import re
from pathlib import Path


def main():
    """Read existing scores and print their arithmetic mean in percent."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True)
    args = parser.parse_args()
    values = []
    for dataset in ('okvqa', 'ocrvqa', 'gqa', 'textvqa'):
        path = args.result_dir / dataset / 'llava-v1.5-7b' / f'result-{dataset}.txt'
        match = re.search(r'Accuracy:\s*([0-9.]+)%', path.read_text())
        if not match:
            raise ValueError(f'Missing accuracy in {path}')
        value = float(match.group(1))
        values.append(value)
        print(f'{dataset}: {value:.2f}')
    print(f'source mean (mean of printed scores): {sum(values)/len(values):.2f}')


if __name__ == '__main__':
    main()
