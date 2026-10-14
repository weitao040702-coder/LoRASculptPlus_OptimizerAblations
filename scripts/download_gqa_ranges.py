#!/usr/bin/env python3
"""Download a large GQA ZIP in fixed HTTP ranges and verify its size."""
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
SIZE = 21_817_965_542
CHUNK = 256 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "assets/data/upstream/downloads/gqa-images.zip"
parts = ROOT / "assets/data/upstream/downloads/gqa-images.parts"
parts.mkdir(exist_ok=True)

def fetch(start: int) -> Path:
    end = min(start + CHUNK, SIZE) - 1
    part = parts / f"{start:012d}-{end:012d}.part"
    expected = end - start + 1
    if not part.is_file() or part.stat().st_size != expected:
        tmp = part.with_suffix(".tmp")
        subprocess.run([
            "curl", "--fail", "--location", "--retry", "20",
            "--retry-all-errors", "--retry-delay", "5",
            "--connect-timeout", "60", "--max-time", "0",
            "--silent", "--show-error", "--range", f"{start}-{end}",
            "--output", str(tmp), URL,
        ], check=True)
        if tmp.stat().st_size != expected:
            raise RuntimeError(f"bad range {start}-{end}: {tmp.stat().st_size}")
        tmp.replace(part)
    return part

starts = list(range(0, SIZE, CHUNK))
with ThreadPoolExecutor(max_workers=8) as pool:
    for index, part in enumerate(pool.map(fetch, starts), 1):
        print(f"completed {index}/{len(starts)} {part.name}", flush=True)

with out.open("wb") as dst:
    for start in starts:
        end = min(start + CHUNK, SIZE) - 1
        part = parts / f"{start:012d}-{end:012d}.part"
        with part.open("rb") as src:
            while block := src.read(16 * 1024 * 1024):
                dst.write(block)

if out.stat().st_size != SIZE:
    raise RuntimeError(f"bad final size: {out.stat().st_size}")
print(f"complete: {out}")
