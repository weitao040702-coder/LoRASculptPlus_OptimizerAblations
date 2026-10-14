# LoRASculpt+ 流形优化代码包说明

打包日期：2026-10-13。代码来自已完成 rank=16/32/64 实验的 `LoRASculptPlus_OptimizerAblations`，保持源文件原样。压缩包只包含代码、构建配置、说明文档和文件校验清单，不包含数据、模型、checkpoint、结果、日志、缓存或 Python 环境。

## 1. 用途与方法边界

方法名称：**masked AB-manifold retraction**。

在已有 LoRASculpt+ 基础上关闭原正交 loss，保留 SRR、LLM CMR、connector CMR；使用普通 AdamW，每次成功 optimizer step 后先回缩 A/B，再重新应用固定 SRR mask。A/B 都训练，未冻结 A；LLM 和 connector 的 LoRA 都执行回缩。

这是欧氏 AdamW 加回缩后处理，没有切空间梯度投影或动量向量运输。SRR 会破坏严格正交，不能描述成严格满足 Stiefel 约束的 Riemannian Adam。

## 2. 代码包布局

解压后的顶层目录为 `LoRASculptPlus_Manifold_Code`。

| 文件/目录 | 作用 |
|---|---|
| `llava/model/ab_manifold.py` | scaled polar 回缩、小 Gram 特征分解、范数及正交残差 |
| `llava/model/lorasculpt_plus.py` | LLM/connector 混合 rank LoRA 构造与配置 |
| `llava/model/builder.py` | 正确识别 ABManifold adapter、恢复混合 rank 并加载视觉塔 |
| `llava/model/` 中其他随包文件 | LLaVA backbone、CLIP 和 connector 必需依赖 |
| `llava/train/LoRASculptPlusOptimizerTrainer.py` | ABManifoldLoRASculptPlus、关闭正交 loss、AdamW 与 post-step 回缩 |
| `llava/train/LoRASculptPlus_Trainer.py` | 基础 SRR/CMR 和成功 optimizer step 后的 hook |
| `llava/train/train.py`、`train_mem.py` | 训练入口与 FlashAttention 路径 |
| `llava/train/` 中其他随包文件 | trainer 注册、sampler、进度和训练依赖 |
| `llava/eval/` | 四项上游推理和评分依赖，不含 IconQA evaluator |
| `llava/optim/muon_optimizer.py` | 共享 trainer/test 文件的静态 import 依赖；不是本包要求执行的方法 |
| `scripts/train_optimizer_config.sh` | 单 GPU 正式训练参数 |
| `scripts/run_one_optimizer_experiment.sh` | 单 rank 训练 wrapper、日志和状态 |
| `scripts/eval_one_upstream.sh` | 四项上游评测 wrapper |
| `scripts/v1_5/eval/upstream/eval_*.sh` | OKVQA/OCRVQA/GQA/TextVQA 四个脚本 |
| `scripts/verify_upstream_eval.py` | 检查评测图像和 annotations |
| `scripts/convert_gqa_for_eval.py`、`score_gqa_accuracy.py` | GQA 格式转换与 accuracy fallback |
| 其他随包进度脚本 | 训练/评测完成记录、失败记录和进度渲染 |
| `tests/test_optimizer_ablations.py` | 含流形数学测试；按下面命令只调用流形测试 |
| `pyproject.toml`、原 `README.md` | 原项目构建配置及构建时 README 依赖 |
| `setup_env.sh` | 原集群环境入口，新服务器需适配 |
| `MANIFOLD_CODE_README.md` | 本说明 |
| `CODE_FILES.sha256` | 包内所有其他文件的 SHA-256 |

未包含多方法自动调度脚本、占卡脚本或历史 sbatch，避免接收方误启动其他实验。共享 Python 文件仍定义其他 trainer，保留是为了不改动已验证源代码；只选择 `ABManifoldLoRASculptPlus`。

## 3. 外部必需资产（代码包没有这些文件）

需要另行准备：

```text
assets/models/llava-v1.5-7b/
assets/models/clip-vit-large-patch14-336/
assets/data/iconqa/iconqa_txt-train-10000k-seed0.json
assets/data/iconqa/raw/data/iconqa_data/
assets/data/eval/okvqa/
assets/data/eval/ocrvqa/
assets/data/eval/gqa/
assets/data/eval/textvqa/
```

必须使用原固定 10000 条训练样本，而不是重新随机抽样。四项评测分别为 OKVQA 5046、deterministic OCRVQA 5000、GQA testdev-balanced 12578、TextVQA 5000；包含必要问题文件、图像和原始 annotations。IconQA 只训练，不评测。

本地 GQA 历史结果使用 accuracy fallback。现有脚本会根据 scene-graph evaluator 文件是否存在选择分支；迁移后必须检查实际评分路径，不能静默切换协议。OCRVQA 必须继续原固定子集。

## 4. 环境要求

已运行环境的主要版本：Python 3.10，torch 2.1.2+cu118，torchvision 0.16.2+cu118，transformers 4.37.2，peft 0.4.0，accelerate 0.21.0，deepspeed 0.12.6，numpy 1.26.4，flash-attn 2.5.8，bitsandbytes 0.41.0。

pyproject 中部分依赖没有 pin，因此不要认为直接 `pip install -e .` 就能得到同一环境。先安装匹配 CUDA/PyTorch，再按版本安装训练依赖并编译兼容的 flash-attn。正式流形训练未启用 DeepSpeed，但基础 trainer 有 import，所以仍需包可导入。

原 setup_env.sh 默认使用项目 `.conda/bin` 并尝试加载 WFU CUDA module；包内没有 `.conda`，新服务器须改为自己激活的环境。不要在未准备环境时盲目 source 后就开始训练。

## 5. 解压与完整性检查

```bash
tar -xzf LoRASculptPlus_Manifold_Code_20261013.tar.gz
cd LoRASculptPlus_Manifold_Code
sha256sum -c CODE_FILES.sha256
```

新服务器完成环境与 assets 布置后：

```bash
export LORASCULPT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
python scripts/verify_upstream_eval.py --eval-root "$PWD/assets/data/eval"
python -m py_compile llava/model/ab_manifold.py llava/train/LoRASculptPlusOptimizerTrainer.py
PYTHONPATH="$PWD:$PWD/tests" python - <<'PY'
from test_optimizer_ablations import (
    test_scaled_retraction_preserves_norm_and_makes_factors_orthogonal,
    test_zero_initialized_B_is_left_unchanged,
)
test_scaled_retraction_preserves_norm_and_makes_factors_orthogonal()
test_zero_initialized_B_is_left_unchanged()
print('AB_MANIFOLD_TESTS=PASS')
PY
```

## 6. 单 GPU smoke 与正式训练

先通过调度器或服务器管理员分配一张 GPU，沿用分配到的 CUDA_VISIBLE_DEVICES。独占服务器才由使用者自行选择 GPU。未设置该变量时评测脚本可能默认多个设备。

下面输出名称必须是新路径；不要覆盖历史结果：

```bash
export OPTAB_METHOD=ab_manifold
export OPTAB_METHOD_LABEL=ABManifold
export OPTAB_RANK=32
export OPTAB_GPU="${CUDA_VISIBLE_DEVICES:-assigned-single-gpu}"
export OPTAB_OUTPUT="$PWD/checkpoints/ABManifold-new-smoke-r32"
export OPTAB_PROGRESS_FILE="$PWD/progress/new-smoke-r32.json"
export OPTAB_CHECKPOINT="$OPTAB_OUTPUT"
export OPTAB_LOG="$PWD/logs/new-smoke-r32.out"
export OPTAB_MAX_STEPS=2
export ORTHO_LAMBDA=0
export CMR_LLM_LAMBDA=1e-3 CMR_CONNECTOR_LAMBDA=1e-5
export STEP_THRESHOLD=100 AB_PRESERVE_RATIO=0.1 OMEGA=1.0
mkdir -p "$PWD/logs"
test ! -e "$OPTAB_OUTPUT" || { echo 'Output already exists'; exit 1; }
bash scripts/train_optimizer_config.sh > "$OPTAB_LOG" 2>&1
```

检查 finite loss、actual optimizer=AdamW、参数更新、回缩记录、scheduler 推进、checkpoint 保存与加载。两步不能验证 step100 后的 SRR，还应在单独测试中覆盖 warmup 激活。

正式训练时 `unset OPTAB_MAX_STEPS`，rank 分别设为 16/32/64，并为每次设置新的 output/progress/log。脚本默认：3 epochs，batch1×accumulation16=global16；LLM LR=2e-4、connector LR=2e-5；两类 alpha64，connector rank128；cosine，warmup0.03，seed/data_seed42；bf16/tf32/gradient checkpointing；1875 optimizer steps。

回缩使用 FP32 r×r Gram eigen decomposition。近零 B 跳过；特征值 clipping/秩亏会导致范数与正交近似误差；回缩后 SRR 进一步改变结构。日志残差不能解读为严格正交证据。

单 GPU 训练历史耗时约 rank16 5.25h、rank32 5.67h、rank64 6.74h（A100 80GB PCIe），新服务器需要实测。每个 rank 的四项评测历史约0.92h。

## 7. 四项评测

先准备一个新 checkpoint，或另行迁入已有完整 adapter 目录。必须包含 `adapter_model.bin`、`adapter_config.json`、`config.json`、`lorasculpt_plus_config.json` 及原保存的其他配置；backbone/CLIP 仍需外部提供。

配置中 `mm_vision_tower` 等可能保留旧绝对路径，须在新副本中修正并记录变更。混合 rank config 不能丢。

```bash
MANIFOLD_CKPT="$PWD/checkpoints/ABManifold-new-r16"
MANIFOLD_EVAL_OUT="$PWD/evaluation/ABManifold-new-r16"
MANIFOLD_EVAL_META="$MANIFOLD_EVAL_OUT/evaluation_metadata.txt"
mkdir -p "$MANIFOLD_EVAL_OUT"
bash scripts/v1_5/eval/upstream/eval_okvqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
bash scripts/v1_5/eval/upstream/eval_ocrvqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
bash scripts/v1_5/eval/upstream/eval_gqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
bash scripts/v1_5/eval/upstream/eval_textvqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
```

核对四个 result 文件和合并预测行数。训练脚本 save_steps10000 大于正式1875 steps，最终 adapter 不是完整 optimizer/scheduler/SRR-mask 续训 checkpoint；后续需要可靠恢复时，在新版本明确加入周期保存和断点测试。

## 8. 接收方须知

这是代码包，不是开箱即跑的完整模型/数据包；本次没有在新服务器安装或执行实验。原服务器源代码未修改。详细科研结论、已有分数、后续 support-aware 流形与方向/幅度分离建议见之前单独的 `LoRASculptPlus_MANIFOLD_HANDOFF.md`，该长文未重复塞入代码包。

包内保留共享 trainer 的 Muon 静态依赖，不应选择 Muon trainer，不执行其他方法。后续仅修改流形机制时使用独立版本目录，记录 seed、代码版本、SRR/CMR、ORTHO_LAMBDA=0、残差、GPU/时间和实际评测协议。
