# OrbitStiefel: preliminary multimodal LoRA experiments

This repository contains a research extension of LoRASculpt for adapting LLaVA-v1.5-7B to IconQA while measuring retention on four visual question answering benchmarks. OrbitStiefel parameterizes LoRA directions using QR-orthogonalized sparse generators and trainable scales, protects approximate leading singular subspaces of the base model, and optionally sparsifies connector scales. This is a milestone, not a claim of a finished benchmark reproduction.

## Preliminary results

The table reports accuracy (%) on four upstream benchmarks after IconQA fine-tuning. The source mean is the arithmetic mean of the four scores. **Bold** marks the highest value within each rank and metric; all values are the recorded results (seed 42).

| LLM rank | Method | OKVQA | OCRVQA | GQA | TextVQA | Source mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | Local baseline | 57.60 | **63.60** | 60.77 | **57.04** | **59.75** |
| 16 | OrbitStiefel | **57.66** | 62.70 | **61.35** | 56.86 | 59.64 |
| 32 | Local baseline | 57.15 | **63.45** | 61.07 | **57.73** | 59.85 |
| 32 | OrbitStiefel | **58.01** | 62.90 | **61.59** | 57.07 | **59.89** |
| 64 | Local baseline | 57.51 | **63.15** | 61.11 | **57.36** | 59.78 |
| 64 | OrbitStiefel | **58.18** | 62.95 | **61.69** | 57.08 | **59.97** |

OrbitStiefel has a higher source mean than the local baseline at ranks 32 and 64, and a lower mean at rank 16. GQA improves over the baseline at all three ranks, while OCRVQA and TextVQA are lower for OrbitStiefel. These are preliminary single-seed results; the small differences do not establish statistical significance. Training and evaluation use the same locally reconstructed data across the main comparison, but execution configurations differ. IconQA target accuracy has not been evaluated. The OCRVQA sample is a reproducible local 2,000-question reconstruction, not the original authors' undisclosed sample. Earlier ABManifold/Muon results in `results.md` use a different protocol and should not be compared directly.

The underlying scores and unrounded means are in `milestone_results/`; reproduce their existing summary with `python3 scripts/milestone_results.py` (less than a second). Means are displayed as recorded, so rounding may differ by 0.01 when calculated from displayed scores.

## Repository layout

- `llava/model/orbit_stiefel.py` and `llava/train/OrbitStiefel_Trainer.py`: method and trainer.
- `scripts/`: data preparation, training, evaluation, and result summaries.
- `configs/`: method and data-protocol settings.
- `milestone_results/`: recorded baseline and method scores.
- `tests/`: mathematical and training/export tests.

Clone this repository and enter its root before running the commands below:

```bash
git clone https://github.com/weitao040702-coder/LoRASculptPlus_OptimizerAblations.git
cd LoRASculptPlus_OptimizerAblations
```

## Environment

Experiments used Python 3.10, PyTorch 2.7.1/CUDA 12.8, Transformers 4.37.2, PEFT 0.4.0 and Accelerate 0.21.0. Use a compatible NVIDIA driver, one or two sufficiently large GPUs, `curl`, `unzip`, and ample disk space (the original assets occupied well over 100 GB). The commands below create an isolated environment; installation on a fresh machine has not yet been verified.

```bash
python3.10 -m venv .venv_new_scheme
source .venv_new_scheme/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-milestone.txt
pip install -e . --no-deps
source setup_env.sh
```

The dependency file records packages used by this project. `pyproject.toml` also has older historical dependency pins; avoid installing those over the explicit PyTorch version. Allow tens of minutes for installation depending on network speed.

## Data and models

Sources: [IconQA](https://iconqa2021.github.io/), [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b), [CLIP ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336), [OK-VQA](https://okvqa.allenai.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [TextVQA](https://textvqa.org/dataset/), [OCR-VQA](https://ocr-vqa.github.io/). LLaVA-formatted GQA and TextVQA question files are supplied by the LLaVA evaluation archive in the download script. Follow each dataset and model's license and access terms.

The download script records exact source URLs, downloads the base models and datasets and extracts them under `assets/`. It uses mirrors for some archives. Downloads and extraction can take many hours and may fail when a source or mirror is unavailable; check errors and rerun after resolving them. The script's final legacy verification expects a different OCR subset, so use the milestone-specific preparation and verification below instead.

```bash
python scripts/download_public_assets.py
mkdir -p assets/data/upstream/downloads assets/data/eval/gqa/data
curl -fL --retry 5 -o assets/data/upstream/downloads/ocrvqa-official-dataset.json https://huggingface.co/datasets/nick121381/ocr_vqa_json/resolve/f6f6a8246fa846768bef63bbe7606e0ad1810162/dataset.json
curl -fL --retry 5 -o assets/data/upstream/downloads/gqa-questions.zip https://huggingface.co/datasets/zrchen03/llava_eval/resolve/main/gqa/questions1.2.zip
unzip -j -o assets/data/upstream/downloads/gqa-questions.zip testdev_balanced_questions.json -d assets/data/eval/gqa/data
python scripts/prepare_milestone_iconqa.py --icon-root assets/data/iconqa/raw/data/iconqa_data --output assets/data/iconqa/iconqa_txt-train-10000-seed42-paper.json
python scripts/prepare_milestone_eval.py --eval-root assets/data/eval --ocr-metadata assets/data/upstream/downloads/ocrvqa-official-dataset.json
python scripts/verify_upstream_eval.py --eval-root assets/data/eval
```

Expected checks: IconQA 10,000 records and SHA-256 `636ef6b88ef780bb82b26fb6f0a573496b0f91d340b406d280e222c90fba1199`; OCRVQA 2,000 records and SHA-256 `9a45d9928cbe35d7a89d0dcd099be67e5c244465ace4d8335ad2be6a057496e9`; OKVQA 5,046, GQA 12,578, TextVQA 5,000. The preparation and verification commands were tested with the original local data, but downloading and extracting all archives on a clean machine remains untested. The OCRVQA subset is locally reconstructed and is not claimed to match the authors' undisclosed IDs. `scripts/prepare_upstream_eval.py` defaults to an older 5,000-question OCRVQA protocol: do not use it as the milestone OCR preparation entry point.

## Published milestone weights

The six selected inference adapters (two methods × three ranks) are available in the [milestone weights release](https://github.com/weitao040702-coder/LoRASculptPlus_OptimizerAblations/releases/tag/milestone-weights-v1). Their `adapter_model.bin` hashes match the recorded completed runs. They require the LLaVA and CLIP base models from the sources above. They do not include optimizer state and cannot be used to resume training.

From a fresh clone, download and verify the six selected archives, then extract them under `checkpoints/`:

```bash
bash scripts/download_milestone_weights.sh
```

The script refuses to overwrite any existing checkpoint and adjusts each extracted configuration to use the local CLIP path. Download time depends on bandwidth. Individual archives and `SHA256SUMS` are also available on the release page. See the evaluation commands below to evaluate these weights directly; training can be skipped. The older [research asset archive](https://huggingface.co/datasets/weitao040702/LoRASculptPlus_OptimizerAblations) contains other project data, but its upload was incomplete. A fresh-machine end-to-end evaluation has not yet been verified.

## Training

For each rank, train the new method as follows. These examples use one GPU with batch 4 and accumulation 4; the original rank-16/32 runs used two GPUs (batch 4, accumulation 2), while rank 64 used one. All use global batch 16, seed 42 and three epochs. Set `CUDA_VISIBLE_DEVICES` to an available GPU, and run from the repository root.

```bash
export CUDA_VISIBLE_DEVICES=0 OPTAB_WORLD_SIZE=1 OPTAB_BATCH_SIZE=4 OPTAB_GRAD_ACCUM=4
export OPTAB_GRAD_CHECKPOINTING=False TORCHDYNAMO_DISABLE=1
for rank in 16 32 64; do
  OPTAB_RANK="$rank" OPTAB_TAG="OrbitStiefel-paper-seed42-r${rank}" bash scripts/train_new_scheme.sh
done
```

For the local original-method control:

```bash
export LORASCULPT_ROOT="$PWD" OPTAB_METHOD=original OPTAB_PYTHON="$PWD/.venv_new_scheme/bin/python"
export OPTAB_WORLD_SIZE=1 OPTAB_BATCH_SIZE=4 OPTAB_GRAD_ACCUM=4 OPTAB_GRAD_CHECKPOINTING=True
for rank in 16 32 64; do
  export OPTAB_RANK="$rank" OPTAB_OUTPUT="$PWD/checkpoints/Original-paper-seed42-r${rank}"
  export OPTAB_PROGRESS_FILE="$OPTAB_OUTPUT/baseline-progress.json"
  bash scripts/train_original_baseline.sh
done
```

Each completed run should reach optimizer step 1,875 and produce `adapter_model.bin`, `adapter_config.json`, model configuration and training state. The trainer refuses to overwrite an existing new-method output; use a fresh tag. Runs are not guaranteed bitwise identical across hardware, GPU counts or recovery. The recorded OrbitStiefel trainer runtimes were approximately 1.8, 2.6 and 2.8 hours for ranks 16, 32 and 64 respectively; original-baseline runs had different GPU/recovery conditions. These are historical trainer-reported times, not guarantees for another machine.

## Evaluation and results

With prepared data and completed model adapters, evaluate all six runs on the four recorded benchmarks:

```bash
export CUDA_VISIBLE_DEVICES=0 LORASCULPT_ROOT="$PWD"
for method in Original-paper-seed42 OrbitStiefel-paper-seed42; do
  for rank in 16 32 64; do
    checkpoint="$PWD/checkpoints/${method}-r${rank}"
    result="$PWD/eval-output/${method}-r${rank}"
    mkdir -p "$result"
    for dataset in okvqa ocrvqa gqa textvqa; do
      bash "scripts/v1_5/eval/upstream/eval_${dataset}.sh" "$checkpoint" "$result" "$result/summary.txt"
    done
    python scripts/summarize_milestone_eval.py --result-dir "$result"
  done
done
python scripts/milestone_results.py
```

Each evaluator saves predictions and a `result-*.txt` in the corresponding dataset directory. The final command prints the **recorded historical results** in `milestone_results/`; it does not score a newly run model. The per-run summarizer reads the new `result-*.txt` files; its mean is calculated from displayed two-decimal scores and can differ by 0.01 from the recorded historical mean because of rounding. GPU evaluation runtime depends on hardware, data loading and concurrent workloads; a clean-machine total has not been measured. No figure is used in this milestone. The source mean averages the four benchmark percentages, with no IconQA target metric.

The method is implemented in `llava/model/orbit_stiefel.py` and `llava/train/OrbitStiefel_Trainer.py`; evaluators are in `llava/eval/` and `scripts/v1_5/eval/upstream/`. For adapters moved from another machine, their `config.json` can contain an absolute `mm_vision_tower` path; set it to the local `assets/models/clip-vit-large-patch14-336` before evaluation. `OPTAB_OUTPUT`, `OPTAB_TRAIN_JSON` and `ORBIT_STIEFEL_CONFIG` override paths. Full end-to-end reproduction on a fresh machine has not yet been verified.

## Attribution

This project builds on [LoRASculpt](https://github.com/LiangJian24/LoRASculpt) and [LLaVA](https://github.com/haotian-liu/LLaVA). Please also cite the original model and datasets when using their artifacts.
