#!/usr/bin/env python3
"""Rebuild the recorded 10,000-example IconQA training subset."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_iconqa_paper import build_records

EXPECTED = '636ef6b88ef780bb82b26fb6f0a573496b0f91d340b406d280e222c90fba1199'


def main():
    """Convert raw IconQA records and reject a mismatched training subset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--icon-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.icon_root
    records = build_records(root, json.loads((root / 'pid_splits.json').read_text()),
                            json.loads((root / 'problems.json').read_text()), 10000, 42)
    payload = json.dumps(records, ensure_ascii=False, indent=2).encode()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED:
        raise ValueError(f'Training data fingerprint mismatch: {digest}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise FileExistsError(f'Refusing to overwrite differing file: {args.output}')
    args.output.write_bytes(payload)
    print(f'IconQA: {len(records)} samples; sha256={digest}; output={args.output}')


if __name__ == '__main__':
    main()
