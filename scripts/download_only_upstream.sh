#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/assets/data/upstream/downloads"
LOG="$ROOT/logs/upstream-download-only.log"
mkdir -p "$OUT" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

download() {
  local name="$1" url="$2" expected="$3"
  local path="$OUT/$name"
  echo "[$(date)] START $name"
  if [[ -f "$path" && "$(stat -c '%s' "$path")" == "$expected" ]]; then
    echo "[$(date)] SKIP COMPLETE $name"
    return 0
  fi
  curl --fail --location --retry 30 --retry-all-errors --retry-delay 10 \
    --connect-timeout 60 --max-time 0 --continue-at - --silent --show-error \
    --output "$path" "$url"
  local actual
  actual=$(stat -c '%s' "$path")
  if [[ "$actual" != "$expected" ]]; then
    echo "[$(date)] FAILED SIZE $name actual=$actual expected=$expected" >&2
    return 1
  fi
  echo "[$(date)] DONE $name bytes=$actual"
}

download "gqa-images.zip" \
  "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" 21817965542 &
P1=$!
download "textvqa-train-val-images.zip" \
  "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" 7072297970 &
P2=$!
download "val2014.zip" \
  "http://images.cocodataset.org/zips/val2014.zip" 6655761201 &
P3=$!
download "ocr_vqa.zip" \
  "https://huggingface.co/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip?download=true" 8931303798 &
P4=$!
download "TextVQA_0.5.1_val.json" \
  "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json" 3116162 &
P5=$!
download "mscoco_val2014_annotations.json.zip" \
  "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip" 395495 &
P6=$!
download "OpenEnded_mscoco_val2014_questions.json.zip" \
  "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip" 116271 &
P7=$!

status=0
for p in "$P1" "$P2" "$P3" "$P4" "$P5" "$P6" "$P7"; do
  wait "$p" || status=1
done
echo "[$(date)] ALL DOWNLOADS status=$status"
exit "$status"
