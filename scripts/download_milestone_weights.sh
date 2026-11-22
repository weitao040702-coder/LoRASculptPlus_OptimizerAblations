#!/usr/bin/env bash
# Download and verify published inference adapters into checkpoints/.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="https://github.com/weitao040702-coder/LoRASculptPlus_OptimizerAblations/releases/download/milestone-weights-v1"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$ROOT/checkpoints"
curl -fL --retry 5 --retry-all-errors -o "$WORK/SHA256SUMS" "$BASE/SHA256SUMS"
for method in Original-paper-seed42 OrbitStiefel-paper-seed42; do
  for rank in 16 32 64; do
    name="${method}-r${rank}"
    if [[ -e "$ROOT/checkpoints/$name" ]]; then
      echo "Refusing to overwrite existing checkpoint: $ROOT/checkpoints/$name" >&2
      exit 1
    fi
  done
done
for method in Original-paper-seed42 OrbitStiefel-paper-seed42; do
  for rank in 16 32 64; do
    name="${method}-r${rank}"
    file="$name.tar"
    curl -fL --retry 5 --retry-all-errors -o "$WORK/$file" "$BASE/$file"
    (cd "$WORK" && grep -F "  $file" SHA256SUMS | sha256sum --check --status)
    tar -xf "$WORK/$file" -C "$ROOT/checkpoints"
    rm "$WORK/$file"
    # The original checkpoint references its training machine's CLIP path.
    python3 - "$ROOT/checkpoints/$name/config.json" "$ROOT/assets/models/clip-vit-large-patch14-336" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
config = json.loads(path.read_text())
config['mm_vision_tower'] = str(Path(sys.argv[2]).resolve())
path.write_text(json.dumps(config, indent=2) + '\n')
PY
    echo "Verified and extracted $name"
  done
done
