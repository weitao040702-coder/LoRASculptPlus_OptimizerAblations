# OrbitStiefel: preliminary multimodal LoRA experiments

This repository contains a research extension of LoRASculpt for adapting LLaVA-v1.5-7B to IconQA while measuring retention on four visual question answering benchmarks. OrbitStiefel parameterizes LoRA directions using QR-orthogonalized sparse generators and trainable scales, protects approximate leading singular subspaces of the base model, and optionally sparsifies connector scales. See [NEW_SCHEME.md](NEW_SCHEME.md). This is a milestone, not a claim of a finished benchmark reproduction.

## Preliminary results

Four-dataset source mean (percentage, seed 42, higher is better):

| LLM rank | Local baseline | OrbitStiefel | No connector sparsity |
| --- | ---: | ---: | ---: |
| 16 | 59.75 | 59.64 | 59.66 |
| 32 | 59.85 | 59.89 | 59.78 |
| 64 | 59.78 | 59.97 | 59.88 |

The underlying four scores and unrounded means are in `milestone_results/`. Print the table using `python3 scripts/milestone_results.py` (less than a second). Training and evaluation use the same locally reconstructed data across the main comparison, but execution configurations differ. Only one seed was run; no IconQA target accuracy is reported. OCRVQA is a reproducible local 2,000-question reconstruction, not the original authors' undisclosed sample. Small differences must not be interpreted as established statistical gains. The older ABManifold/Muon results in `results.md` use a different protocol and should not be compared directly.

## Environment

The research environment used Python 3.10, PyTorch 2.7.1 with CUDA 12.8, Transformers 4.37.2, PEFT 0.4.0 and Accelerate 0.21.0. A CUDA-capable NVIDIA GPU and substantial storage are required for 7B experiments. The following creates an environment; driver compatibility and dependent package availability may require adjustment:

```bash
python3.10 -m venv .venv_new_scheme
source .venv_new_scheme/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e . --no-deps
pip install transformers==4.37.2 peft==0.4.0 accelerate==0.21.0 deepspeed==0.16.9 numpy==1.26.4 sentencepiece==0.1.99 tokenizers==0.15.1 shortuuid 'markdown2[all]' scikit-learn==1.2.2 einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 pillow requests tqdm safetensors huggingface_hub
source setup_env.sh
```

This installation recipe has not yet been verified on a clean machine. `pyproject.toml` includes historical dependency pins and should not be used to replace the explicit PyTorch installation above.

## Data and models

The scripts expect `assets/models/llava-v1.5-7b`, `assets/models/clip-vit-large-patch14-336`, `assets/data/iconqa/`, and `assets/data/eval/`. Public model sources are [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [CLIP ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336); `scripts/download_public_assets.py` records dataset download sources. Dataset and model licenses apply. The [asset archive](https://huggingface.co/datasets/weitao040702/LoRASculptPlus_OptimizerAblations) is public, but its upload was incomplete and it has not been verified to contain the current OrbitStiefel checkpoints or all data. **A clean-machine end-to-end reproduction is not yet available**; obtain missing inputs separately before running. Do not assume the archived weights correspond to the results below.

The training set is a fixed 10,000-example seed-42 IconQA choose-text sample, not the undisclosed author sample. `scripts/prepare_iconqa_paper.py` defines its conversion; `scripts/auto_paper_pipeline.py` prepares it. OCRVQA provenance and sample hash are in `configs/paper_eval_provenance.json`. See [AUTO_PAPER_PIPELINE.md](AUTO_PAPER_PIPELINE.md) for protocol, execution changes and the GQA accuracy implementation. Preparing all public assets can take many hours and substantial disk space; the downloader and pipeline have not been verified on a fresh machine.

## Training and evaluation

After installing dependencies and preparing all assets, example commands for the new method are:

```bash
export CUDA_VISIBLE_DEVICES=0
export OPTAB_WORLD_SIZE=1 OPTAB_BATCH_SIZE=4 OPTAB_GRAD_ACCUM=4
export OPTAB_GRAD_CHECKPOINTING=False
OPTAB_RANK=64 bash scripts/train_new_scheme.sh
```

Repeat with ranks 16 and 32. Training uses 3 epochs, global batch 16 and 1,875 optimizer steps, and writes an adapter and training state under `checkpoints/`. The original experiment used two GPUs for ranks 16/32 and one for rank 64, with recovery and concurrency details in the protocol document. A new run can differ numerically. Runtime depends heavily on GPU and concurrent jobs; historical ABManifold runs took roughly 5–7 hours each, which is not a measured OrbitStiefel runtime.

To evaluate one exported adapter, set `CHECKPOINT` to its directory and run the same four scripts. Each expects prepared benchmark images and annotations, and writes predictions and `result-*.txt` under the output directory:

```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT="$PWD/checkpoints/OrbitStiefel-v1-seed42-r64"
export RESULT_DIR="$PWD/eval-output/r64"
mkdir -p "$RESULT_DIR"
for dataset in okvqa ocrvqa gqa textvqa; do
  bash "scripts/v1_5/eval/upstream/eval_${dataset}.sh" "$CHECKPOINT" "$RESULT_DIR" "$RESULT_DIR/summary.txt"
done
```

The new method implementation lives in `llava/model/orbit_stiefel.py` and `llava/train/OrbitStiefel_Trainer.py`; evaluation is in `llava/eval/` and `scripts/v1_5/eval/upstream/`. Tests are in `tests/`. Use `OPTAB_OUTPUT`, `OPTAB_TRAIN_JSON` and `ORBIT_STIEFEL_CONFIG` to override paths/configuration; do not store credentials in the repository. Commands above document existing entry points and have not yet been validated end-to-end on a clean machine. No plot is used in the current milestone.
