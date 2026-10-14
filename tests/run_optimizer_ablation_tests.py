#!/usr/bin/env python3
"""Run the optimizer-ablation unit tests without requiring pytest."""

from __future__ import annotations

import inspect

import test_optimizer_ablations


def main():
    tests = [
        value for name, value in vars(test_optimizer_ablations).items()
        if name.startswith("test_") and inspect.isfunction(value)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"{len(tests)} optimizer-ablation tests passed")


if __name__ == "__main__":
    main()
