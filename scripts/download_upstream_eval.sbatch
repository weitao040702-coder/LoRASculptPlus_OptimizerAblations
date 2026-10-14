#!/usr/bin/env bash
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-upstream-data
#SBATCH --output=/deac/csc/yangGrp/qianr/LoRASculpt/logs/lorasculpt-upstream-data-%j.out
#SBATCH --error=/deac/csc/yangGrp/qianr/LoRASculpt/logs/lorasculpt-upstream-data-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source ./setup_env.sh
mkdir -p logs assets/data/upstream/downloads assets/data/eval

DOWNLOAD_ROOT="$LORASCULPT_ROOT/assets/data/upstream/downloads"
EVAL_ROOT="$LORASCULPT_ROOT/assets/data/eval"

download() {
  local url="$1"
  local output="$2"
  echo "downloading: $url -> $output"
  curl --fail --location --retry 20 --retry-all-errors --retry-delay 10 \
    --connect-timeout 60 --max-time 0 --continue-at - \
    --output "$output" "$url"
}

check_size() {
  local path="$1"
  local expected="$2"
  local actual
  actual=$(stat -c '%s' "$path")
  if [[ "$actual" != "$expected" ]]; then
    echo "size check failed: $path actual=$actual expected=$expected" >&2
    return 1
  fi
  echo "size check passed: $path ($actual bytes)"
}

echo "job=$SLURM_JOB_ID host=$(hostname)"
df -h "$LORASCULPT_ROOT"

# The small LLaVA annotation archive was already probed successfully. Keep a
# copy in the project so the preparation is self-contained.
if [[ -s /tmp/llava-eval-probe.bin && ! -s "$DOWNLOAD_ROOT/llava-eval.zip" ]]; then
  cp /tmp/llava-eval-probe.bin "$DOWNLOAD_ROOT/llava-eval.zip"
fi

download "https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip" \
  "$DOWNLOAD_ROOT/gqa-eval.zip"
check_size "$DOWNLOAD_ROOT/gqa-eval.zip" 821885879
download "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
  "$DOWNLOAD_ROOT/gqa-images.zip"
check_size "$DOWNLOAD_ROOT/gqa-images.zip" 21817965542
download "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json" \
  "$DOWNLOAD_ROOT/TextVQA_0.5.1_val.json"
check_size "$DOWNLOAD_ROOT/TextVQA_0.5.1_val.json" 3116162
download "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
  "$DOWNLOAD_ROOT/textvqa-train-val-images.zip"
check_size "$DOWNLOAD_ROOT/textvqa-train-val-images.zip" 7072297970
download "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip" \
  "$DOWNLOAD_ROOT/mscoco_val2014_annotations.json.zip"
download "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip" \
  "$DOWNLOAD_ROOT/OpenEnded_mscoco_val2014_questions.json.zip"
# The HTTPS endpoint presents a certificate for a different host on this
# cluster; the same official S3 object is available over HTTP.
download "http://images.cocodataset.org/zips/val2014.zip" \
  "$DOWNLOAD_ROOT/val2014.zip"
check_size "$DOWNLOAD_ROOT/val2014.zip" 6645013297
download "https://huggingface.co/datasets/qnguyen3/ocr_vqa/resolve/main/ocr_vqa.zip?download=true" \
  "$DOWNLOAD_ROOT/ocr_vqa.zip"
check_size "$DOWNLOAD_ROOT/ocr_vqa.zip" 8931303798

echo "all downloads complete"
du -sh "$DOWNLOAD_ROOT"

mkdir -p "$EVAL_ROOT/gqa/data" "$EVAL_ROOT/textvqa"
unzip -q -o "$DOWNLOAD_ROOT/llava-eval.zip" -d "$EVAL_ROOT"
unzip -q -o "$DOWNLOAD_ROOT/gqa-eval.zip" -d "$EVAL_ROOT/gqa/data"
unzip -q -o "$DOWNLOAD_ROOT/mscoco_val2014_annotations.json.zip" -d "$EVAL_ROOT/okvqa"
unzip -q -o "$DOWNLOAD_ROOT/OpenEnded_mscoco_val2014_questions.json.zip" -d "$EVAL_ROOT/okvqa"
cp -f "$DOWNLOAD_ROOT/TextVQA_0.5.1_val.json" "$EVAL_ROOT/textvqa/TextVQA_0.5.1_val.json"

mkdir -p "$EVAL_ROOT/gqa/data/images" "$EVAL_ROOT/textvqa/train_images" \
  "$EVAL_ROOT/okvqa/val2014" "$EVAL_ROOT/ocrvqa/raw"
unzip -q -o "$DOWNLOAD_ROOT/gqa-images.zip" -d "$EVAL_ROOT/gqa/data"
unzip -q -o "$DOWNLOAD_ROOT/textvqa-train-val-images.zip" -d "$EVAL_ROOT/textvqa"
unzip -q -o "$DOWNLOAD_ROOT/val2014.zip" -d "$EVAL_ROOT/okvqa"
unzip -q -o "$DOWNLOAD_ROOT/ocr_vqa.zip" -d "$EVAL_ROOT/ocrvqa/raw"

echo "all archives extracted"
find "$EVAL_ROOT" -maxdepth 3 -type f -printf '%s %p\n' | sort -k2 | tail -n 80

python scripts/prepare_upstream_eval.py --eval-root "$EVAL_ROOT"
python scripts/verify_upstream_eval.py --eval-root "$EVAL_ROOT"
