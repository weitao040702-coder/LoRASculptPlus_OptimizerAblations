#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DOWNLOADS="$ROOT/assets/data/upstream/downloads"
EVAL="$ROOT/assets/data/eval"
mkdir -p "$EVAL/gqa/data/images" "$EVAL/textvqa" "$EVAL/okvqa" "$EVAL/ocrvqa/raw"

echo "[$(date)] extracting evaluator and annotations"
unzip -q -o "$DOWNLOADS/gqa-eval.zip" -d "$EVAL/gqa/data"
unzip -q -o "$DOWNLOADS/llava-eval.zip" -d "$EVAL"
unzip -q -o "$DOWNLOADS/mscoco_val2014_annotations.json.zip" -d "$EVAL/okvqa"
unzip -q -o "$DOWNLOADS/OpenEnded_mscoco_val2014_questions.json.zip" -d "$EVAL/okvqa"
cp -f "$DOWNLOADS/TextVQA_0.5.1_val.json" "$EVAL/textvqa/TextVQA_0.5.1_val.json"

echo "[$(date)] extracting GQA test images only"
python - "$DOWNLOADS/gqa-images.zip" "$EVAL/gqa/data/images" <<'PY'
import json, subprocess, sys
zip_path, out = sys.argv[1:]
names = sorted({json.loads(line)['image'] for line in open('assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl')})
for image in names:
    subprocess.run(['unzip', '-q', '-o', zip_path, 'images/' + image, '-d', out], check=True)
print('GQA images:', len(names))
PY
mv "$EVAL/gqa/data/images/images"/* "$EVAL/gqa/data/images/" 2>/dev/null || true
rmdir "$EVAL/gqa/data/images/images" 2>/dev/null || true

echo "[$(date)] extracting TextVQA images"
unzip -q -o "$DOWNLOADS/textvqa-train-val-images.zip" -d "$EVAL/textvqa"

echo "[$(date)] extracting COCO val2014"
unzip -q -o "$DOWNLOADS/val2014.zip" -d "$EVAL/okvqa"

echo "[$(date)] extracting OCR-VQA"
unzip -q -o "$DOWNLOADS/ocr_vqa.zip" -d "$EVAL/ocrvqa/raw"

echo "[$(date)] converting annotations"
python scripts/prepare_upstream_eval.py --eval-root "$EVAL"
python scripts/verify_upstream_eval.py --eval-root "$EVAL"
echo "[$(date)] preparation complete"
