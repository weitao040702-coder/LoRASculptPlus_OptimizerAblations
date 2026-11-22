#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/automation/orbit-no-connector-sparse-seed42"
LOG="$ROOT/logs/connector-ablation"
PYTHON="$ROOT/.venv_new_scheme/bin/python"
mkdir -p "$WORK" "$LOG"
if tmux has-session -t orbit-no-connector 2>/dev/null; then
  echo 'Session orbit-no-connector already exists; inspect it before restarting.'
  exit 0
fi
tmux new-session -d -s orbit-no-connector -n pipeline "env PAPER_PIPELINE_WORK='$WORK' PAPER_PIPELINE_LOG='$LOG' PAPER_PIPELINE_GPUS=0,1 PAPER_PIPELINE_EVAL_FOLDER=eval_results_paper_gpu01 PAPER_PIPELINE_TAG_PREFIX=OrbitStiefelNoConnectorSparse-paper-seed42 PAPER_PIPELINE_METHOD_CONFIG='$ROOT/configs/orbit_stiefel_no_connector_sparse.json' PAPER_PIPELINE_METHOD_LABEL=OrbitStiefelNoConnectorSparse '$PYTHON' -u '$ROOT/scripts/auto_paper_pipeline.py' >> '$LOG/pipeline.log' 2>&1"
tmux new-window -t orbit-no-connector -n monitor "env PAPER_PIPELINE_WORK='$WORK' PAPER_PIPELINE_LOG='$LOG' PAPER_PIPELINE_EVAL_FOLDER=eval_results_paper_gpu01 PAPER_PIPELINE_TAG_PREFIX=OrbitStiefelNoConnectorSparse-paper-seed42 PAPER_PIPELINE_TITLE='Connector谱稀疏正则关闭消融实时进度' PAPER_PIPELINE_EXEC_NOTE='训练和评测均由GPU0/1双卡并行；seed42；rank16/32/64；3epochs；global batch16。' '$PYTHON' -u '$ROOT/scripts/watch_paper_progress.py' >> '$LOG/monitor.log' 2>&1"
echo "Progress: $WORK/live.md"
