# Optimizer Ablation Results

Only AB-manifold and Muon are trained in this directory. The existing LoRASculpt+ runs are read-only references.
Source Avg = (OKVQA + OCRVQA + GQA + TextVQA) / 4.

| Method | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source Avg | Delta vs Local LoRASculpt+ | Delta vs Paper | Train Time | Eval Time | Peak VRAM | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AB-manifold | 16 | 57.92 | 63.50 | 61.79 | 57.99 | 60.30 | 0.34 | 0.85 | 5.25 h | 0.91 h | 33743 MiB | COMPLETED |
| AB-manifold | 32 | 58.00 | 63.70 | 61.78 | 57.97 | 60.36 | 0.44 | 0.37 | 5.67 h | 0.92 h | 33743 MiB | COMPLETED |
| AB-manifold | 64 | 58.20 | 63.64 | 62.12 | 58.08 | 60.51 | 0.55 | 1.54 | 6.74 h | 0.92 h | 33743 MiB | COMPLETED |
| Muon | 16 | 48.99 | 57.46 | 48.46 | 44.97 | 49.97 | -9.99 | -9.48 | 5.09 h | 0.90 h | 32865 MiB | COMPLETED |
| Muon | 32 | 55.47 | 63.34 | 58.68 | 53.71 | 57.80 | -2.13 | -2.19 | 5.55 h | 0.91 h | 32865 MiB | COMPLETED |
| Muon | 64 | 56.99 | 62.26 | 61.05 | 56.87 | 59.29 | -0.66 | 0.33 | 6.41 h | 0.92 h | 32865 MiB | COMPLETED |

## Benchmark Deltas vs Local LoRASculpt+

| Method | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source Avg |
|---|---:|---:|---:|---:|---:|---:|
| AB-manifold | 16 | 0.52 | 0.08 | 0.57 | 0.20 | 0.34 |
| AB-manifold | 32 | 0.70 | 0.72 | 0.40 | -0.08 | 0.44 |
| AB-manifold | 64 | 0.74 | 0.90 | 0.31 | 0.26 | 0.55 |
| Muon | 16 | -8.41 | -5.96 | -12.76 | -12.82 | -9.99 |
| Muon | 32 | -1.83 | 0.36 | -2.70 | -4.34 | -2.13 |
| Muon | 64 | -0.47 | -0.48 | -0.76 | -0.95 | -0.66 |

## Benchmark Deltas vs TPAMI paper

| Method | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source Avg |
|---|---:|---:|---:|---:|---:|---:|
| AB-manifold | 16 | 0.58 | 1.40 | 0.83 | 0.59 | 0.85 |
| AB-manifold | 32 | 0.77 | -0.80 | 0.92 | 0.59 | 0.37 |
| AB-manifold | 64 | 1.55 | -0.26 | 2.28 | 2.60 | 1.54 |
| Muon | 16 | -8.35 | -4.64 | -12.50 | -12.43 | -9.48 |
| Muon | 32 | -1.76 | -1.16 | -2.18 | -3.67 | -2.19 |
| Muon | 64 | 0.34 | -1.64 | 1.21 | 1.39 | 0.33 |

## Configuration

- Fixed training data: `assets/data/iconqa/iconqa_txt-train-10000k-seed0.json`; IconQA is not evaluated.
- Backbone: LLaVA-v1.5-7B; epochs=3; global batch size=16; per-device batch=1; gradient accumulation=16.
- LLM rank=16/32/64; connector rank=128; LLM and connector alpha=64.
- Learning rates: LLM `2e-4`, connector `2e-5`; weight decay=0; cosine scheduler; warmup ratio=0.03.
- bf16=True; tf32=True; gradient checkpointing=True; max length=2048; seed=42; DeepSpeed=False.
- Original orthogonal loss is disabled with `ORTHO_LAMBDA=0`; SRR and both CMR terms remain enabled.
- AB-manifold uses AdamW plus post-step masked AB-manifold retraction. Muon uses `MuonWithAdamW` with momentum=0.95, Newton-Schulz steps=5, nesterov=True.

## Evaluation Protocol

- Only OKVQA (5046), deterministic OCRVQA (5000), GQA testdev-balanced (12578), and TextVQA (5000) are evaluated; temperature=0.
- GQA uses the local accuracy fallback if the optional scene-graph evaluator is unavailable.
- Differences from the official paper protocol are limited to the deterministic local OCRVQA subset and the documented GQA fallback.
