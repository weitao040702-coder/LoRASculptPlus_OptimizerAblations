# LoRASculpt+ reproduction

This directory is an isolated reproduction workspace for the TPAMI paper
"LoRASculpt: Harmonious Low-Rank Adaptation for Multimodal Large Language
Models" (DOI: 10.1109/TPAMI.2026.3707616). The original LoRASculpt directory is
left untouched. `assets` is a symbolic link to the existing model and dataset
assets; logs, checkpoints, and caches are local to this directory.

## Paper protocol implemented here

- Backbone: LLaVA-v1.5-7B.
- Downstream task: IconQA, a fixed random sample of 10,000 training examples.
- Training: 3 epochs, global batch size 16, LLM learning rate `2e-4`, connector
  learning rate `2e-5`.
- LLM LoRA rank: 32. Connector LoRA rank: 128.
- SRR: after a 100-step warmup, retain the largest 10% of each LLM LoRA A/B
  factor and keep this binary mask fixed. The conference implementation used
  by the original workspace defines the warmup default as 100; the TPAMI paper
  gives the warmup symbol but does not print a numeric value.
- RKH/CMR: only LLM `q_proj`, `k_proj`, and `v_proj`, Frobenius norm,
  `omega=1`, `lambda_llm=1e-3`.
- Connector CMR: all connector LoRA layers, L1 norm,
  `lambda_connector=1e-5`.
- OCE: `R_ortho(B) + R_ortho(A^T)` for every LLM and connector LoRA layer,
  `lambda_ortho=1e-5`.
- `connector_lora_alpha=64` is a compatibility/reproducibility assumption;
  the paper specifies the connector rank but does not disclose a separate
  connector alpha. LLM alpha 64 follows the established local LoRASculpt
  reproduction configuration.

The implemented objective is Eq. (18):

```text
L = L_task + lambda_llm * L_CMR_LLM
          + lambda_connector * L_CMR_connector
          + lambda_ortho * L_ortho
```

The upstream evaluation datasets remain the four paper datasets: OKVQA,
OCRVQA, GQA, and TextVQA. Evaluation should be run only after the Plus
checkpoint has been produced. The existing local evaluation scripts use some
subsets/metric adapters for storage reasons, so their outputs must be labeled
as local reproduction numbers rather than exact paper-table numbers unless the
full official evaluation protocol is used.

## Checks and training

Run the local checks from this directory:

```bash
source setup_env.sh
python tests/run_lorasculpt_plus_tests.py
python -m py_compile llava/model/lorasculpt_plus.py llava/train/LoRASculptPlus_Trainer.py llava/train/train.py llava/model/builder.py
bash -n scripts/v1_5/train/trainconfig_lora.sh scripts/v1_5/train/train_lorasculpt_plus_iconqa10k_l40s.sbatch
```

When the Slurm controller is available, the guarded submission helper runs a
2-step, 4x-L40S smoke test first, then submits formal training and the four
upstream evaluations with `afterok` dependencies:

```bash
bash scripts/submit_lorasculpt_plus.sh
```

Submit the four-GPU experiment with:

```bash
sbatch scripts/v1_5/train/train_lorasculpt_plus_iconqa10k_l40s.sbatch
```

The output directory is under `checkpoints/` and is deliberately separate
from the original LoRASculpt checkpoints.
