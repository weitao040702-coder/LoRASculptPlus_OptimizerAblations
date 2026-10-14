# LoRASculpt+ 流形优化改进：完整科研交接文档

核对日期：2026-10-11（America/New_York）。本文面向接收项目的研究者或实验执行 agent，覆盖已实现的 masked AB-manifold retraction、已有实验、代码位置、迁移与复跑、局限和后续研究路线。只讨论流形优化方向；同目录其他实验不属于本文的继续执行范围。

## 1. 先明确项目现状与边界

本项目在本地 LoRASculpt+ 复现代码上，把原有 A/B 非对角 Gram 正交惩罚替换为每次 AdamW 更新后的 scaled Stiefel 回缩；随后重新应用原 SRR 固定元素掩码。保留 LLM CMR、connector CMR、原有 LoRA target modules、backbone、训练数据及本地四个上游评测协议。

已经完成 LLM rank=16、32、64 三次正式训练，每次 1875 optimizer steps、3 epochs；每个 checkpoint 都完成 OKVQA、OCRVQA、GQA、TextVQA 四项评测。本文整理时逐一核对了三个 `trainer_state.json` 的 step/epoch、三个 adapter 文件、十二个结果文件及十二个合并预测文件的行数。

已有单次实验显示：三个 rank 的四项等权平均都高于已有本地 LoRASculpt+，提升分别为 **0.3425、0.4350、0.5525 个百分点**。这些是 seed=42 的单次结果，不能据此宣称统计显著、严格论文协议下全面领先或已经满足 CVPR 发表要求。

本次交接整理未启动新训练、未申请或释放 GPU、未传输项目，也没有改动原实验代码或结果。文档中的运行命令用于新服务器后续操作，不代表已在新服务器执行。

继续研究的基本规则：

- 原 LoRASculpt+ baseline、已有 ABManifold-r16/r32/r64 及其结果保持只读。
- 后续方法、seed、消融必须使用新的实验目录和输出路径。
- IconQA 是固定训练数据，本项目四项上游评测不包含 IconQA；不要执行 IconQA evaluation。
- 不要直接启动现有“all / pending / parallel phase”批处理脚本：它们可能包含本文范围之外的方法。
- 历史 Slurm job ID 只用于溯源，不能当作当前有效 allocation。新服务器资源需要重新确认。

## 2. 原服务器项目布局与证据文件

原服务器 SSH 登录身份为 `qianr@lovelace.deac.wfu.edu`。相关路径：

| 用途 | 原服务器绝对路径 |
|---|---|
| 实验父目录 | `/deac/csc/yangGrp/qianr/qiushi/mwt` |
| 原始基础项目、共享 assets 和环境 | `/deac/csc/yangGrp/qianr/qiushi/mwt/LoRASculpt` |
| 本地 LoRASculpt+ baseline | `/deac/csc/yangGrp/qianr/qiushi/mwt/LoRASculptPlus` |
| 已完成流形实验代码与输出 | `/deac/csc/yangGrp/qianr/qiushi/mwt/LoRASculptPlus_OptimizerAblations` |
| 本地 baseline 汇总 | `LoRASculptPlus/rank16_32_64_paper_style_results.md` |
| 流形结果及差值的现有汇总 | `LoRASculptPlus_OptimizerAblations/results.md`、`results.csv` |
| 参数、论文参考值及诊断 | `LoRASculptPlus_OptimizerAblations/run_manifest.json` |
| 各 rank 最终状态 | `LoRASculptPlus_OptimizerAblations/progress/ABManifold-r{16,32,64}.json` |
| 现有总进度表 | `LoRASculptPlus_OptimizerAblations/progress/progress.md`、`progress.json`、`progress.tsv` |

以下章节未特别说明的相对路径均相对于 `LoRASculptPlus_OptimizerAblations`。该目录名称包含 OptimizerAblations 是历史组织方式；本文只继承其中 AB-manifold 分支。

### 2.1 最重要的代码文件

| 文件 | 实际职责 |
|---|---|
| `llava/model/ab_manifold.py` | FP32 小 Gram 特征分解、scaled polar 回缩、范数及残差诊断 |
| `llava/train/LoRASculptPlusOptimizerTrainer.py` | `ABManifoldLoRASculptPlus`；AdamW 分组；关闭原正交项；post-step 回缩后 SRR |
| `llava/train/LoRASculptPlus_Trainer.py` | 基础 trainer；SRR 初始化/梯度 hook/参数 mask；CMR；成功 optimizer step 后的扩展 hook |
| `llava/model/lorasculpt_plus.py` | LLM/connector 混合 rank LoRA 注入、冻结非 LoRA 参数、配置读写和布局检查 |
| `llava/train/train.py` | trainer 选择、模型及 adapter 构造、训练入口和最终保存 |
| `llava/train/train_mem.py` | FlashAttention 训练入口 |
| `llava/model/builder.py` | 识别非标准命名 adapter checkpoint、正确恢复混合 rank adapter、合并后评测 |
| `llava/train/progress.py` | 进度状态保存 |
| `scripts/train_optimizer_config.sh` | 本地正式单 GPU 训练参数 |
| `scripts/run_one_optimizer_experiment.sh` | 一次训练的日志、GPU 采样、状态和文件检查 |
| `scripts/eval_one_upstream.sh` | 顺序运行四个上游评测并保存状态/耗时 |
| `scripts/v1_5/eval/upstream/eval_{okvqa,ocrvqa,gqa,textvqa}.sh` | 四项推理、分片合并、样本数检查和评分 |
| `scripts/verify_upstream_eval.py` | 评测数据完整性检查 |
| `tests/test_optimizer_ablations.py` | 包含流形范数/正交及零初始化测试 |

迁移时不能只搬 `ab_manifold.py`：基础 trainer 的 post-step hook、混合 rank 注入和修改后的 builder 都是必要依赖。

## 3. 通俗解释：到底换了什么

LoRA 用两个小矩阵构造对大模型权重的增量。可以把 A 的各行、B 的各列看成低秩更新的若干“方向”。如果这些方向彼此过于相似，多个方向可能重复表达同一件事，有限 rank 的表示效率下降。

原方法在 loss 中加一项“方向相似就扣分”的正交惩罚。这个惩罚与任务 loss、CMR 同时优化，其效果取决于系数和训练过程，并不保证每次更新后方向真的正交。

当前方法先正常训练一步，再主动把 A/B 的方向整理到近似正交的几何结构上，同时尽量保留每个因子的整体范数。随后应用 SRR，让不允许更新的 LLM 元素重新为零。

因此它改变的是**参数更新后的几何处理**，没有换训练集、backbone、四项评测或把 A 冻结。A 和 B 都参与梯度训练，connector 的 A/B 也参与回缩。

注意：整理方向后再置零，会破坏刚整理好的正交结构。因此准确名称为 **masked AB-manifold retraction**；不能声称最终 A/B 严格处在 Stiefel 流形上，也不能称为完整的 Riemannian Adam。

## 4. 原 LoRASculpt+ 在当前代码中的机制

本节描述读取到的本地实现，不把本地实现与论文原作者代码未经核验地等同。

### 4.1 LoRA 与混合 rank

对某个冻结权重 \(W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\)，实际 LoRA 增量为

\[
\Delta W=\frac{\alpha}{r}BA,\qquad
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\quad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
\]

LLM rank 测试 16、32、64，LLM alpha=64；connector rank 固定 128，connector alpha=64。LLM forward scaling 分别为 4、2、1，connector scaling 为 0.5。不同 rank 的结果不能理解成只增加参数数目而完全不改变 forward scaling。

LLM targets 为 `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`，不包括 `lm_head` 和视觉塔。正式配置记录 224 个 LLM LoRA layer，加 connector 的 `mm_projector.0`、`mm_projector.2` 两层，共 226 层。LoRA dropout=0.05，bias=`none`；非 LoRA 参数冻结。

PEFT 0.4.0 没有现代的 `rank_pattern` API，本地通过 `lorasculpt_plus.py` 单独替换 connector 线性层实现混合 rank。不要用普通单 rank PEFT 加载器忽略 connector 配置。

### 4.2 SRR：固定稀疏保留

前 100 个 optimizer steps 不应用固定稀疏 mask。基础 trainer 在 `state.global_step >= 100` 时、下一次前向/反向准备阶段首次生成 mask。

对每个 **LLM** LoRA A/B 参数独立 flatten，取绝对值最大的

\[
n_{\mathrm{keep}}=\max\{1,\lfloor 0.1\,n\rfloor\}
\]

个元素并生成 bool mask。之后 mask 固定，不每步重选。梯度通过 hook 被 mask，optimizer step 后参数再次被 mask。connector LoRA 不受这个 LLM SRR mask 限制。

SRR mask 保存/加载函数在基础 trainer 中；周期 checkpoint 文件名是 `lorasculpt_plus_srr_masks.pt`。正式最终 adapter 目录不能因此自动视作具备完整续训状态，详见恢复训练章节。

### 4.3 CMR：基于冻结权重的加权更新惩罚

本地首先把冻结权重以 Frobenius norm 归一化：

\[
\widehat W=\frac{W_0}{\max(\|W_0\|_F,\varepsilon)},\qquad
R(W_0)=\tanh\!\left(\omega\left|\frac{1}{\log(|\widehat W|+\varepsilon)}\right|\right),
\]

其中 \(\varepsilon=10^{-15}\)、\(\omega=1\)，代码还对 NaN/Inf 做处理。

当前正则实现直接使用 **未乘 alpha/r 的 `B @ A`**：

\[
L_{\mathrm{LLM\text{-}CMR}}=
\sum_{\ell\in\{q,k,v\}}\|R(W_{0,\ell})\odot(B_\ell A_\ell)\|_F,
\]

\[
L_{\mathrm{connector\text{-}CMR}}=
\sum_{\ell\in\mathrm{connector}}\|\operatorname{vec}(R(W_{0,\ell})\odot(B_\ell A_\ell))\|_1.
\]

这里 `{q,k,v}` 表示所有 q_proj/k_proj/v_proj 层；LLM 其他 targets 有 LoRA 和回缩，但当前 LLM CMR 没覆盖它们。保留这一行为才能延续已有本地协议。若今后改成约束缩放后的实际更新或扩大 CMR 作用范围，必须作为单独消融。

### 4.4 被替换掉的原正交项

定义

\[
\operatorname{offdiag}(G)=G-\operatorname{diag}(\operatorname{diag}(G)).
\]

原本地正交项为

\[
L_{\mathrm{ortho}}=\sum_\ell
\left(\|\operatorname{offdiag}(B_\ell^\top B_\ell)\|_F^2
+\|\operatorname{offdiag}(A_\ell A_\ell^\top)\|_F^2\right).
\]

它只惩罚非对角项，不要求各列/行范数相同。原系数 `ORTHO_LAMBDA=1e-5`。

流形分支不仅设置环境变量 `ORTHO_LAMBDA=0`，还覆盖 `compute_custom_regularizers()`，仅计算两项 CMR，第三返回值是零。因此不是在原 loss 上叠加流形；原正交项不参与总 loss：

\[
L=L_{\mathrm{task}}+10^{-3}L_{\mathrm{LLM\text{-}CMR}}
+10^{-5}L_{\mathrm{connector\text{-}CMR}}.
\]

## 5. 当前流形回缩的数学与实现

### 5.1 理想目标

希望每层达到

\[
B^\top B=c_BI_r,\qquad AA^\top=c_AI_r,
\]

\[
c_B=\frac{\|B_{\mathrm{pre}}\|_F^2}{r},\qquad
c_A=\frac{\|A_{\mathrm{pre}}\|_F^2}{r}.
\]

`pre` 指本次 optimizer step 后、回缩前的参数。这些尺度每步重新测量，并非整个训练固定。

本方法比原非对角 loss 更强：除了降低不同方向的相关性，还把理想满秩矩阵的各方向长度均衡。因此性能变化不能只归因于“正交做得更精确”。

### 5.2 实际算法：small-Gram polar

`_scaled_stiefel(matrix, eps=1e-8)` 先 detach 并转换 FP32。若范数非有限或不超过 eps，就跳过；这正确处理零初始化 B，但非有限跳过并不能修复 NaN，训练必须另外检查 finite。

对高矩阵 \(X\in\mathbb{R}^{d\times r}\)：

\[
G=X^\top X=V\operatorname{diag}(\lambda_i)V^\top,
\]

\[
H=V\operatorname{diag}\!\left((\max(\lambda_i,10^{-8}))^{-1/2}\right)V^\top,
\]

\[
X_{\mathrm{new}}=XH\frac{\|X\|_F}{\sqrt r}.
\]

B 直接作为 X；A 以 A.T 作为 X，结束再 transpose 回来。模块还支持宽矩阵的左 Gram 分支。通过 r×r Gram 求特征分解，避免每层每步对完整隐藏维度矩阵进行大型 SVD，也不构造整个 pretrained weight 来回缩。

最后转换回参数 dtype 并 `copy_` 到原 A/B。在满秩、特征值未触及下限、数值精度足够时得到 scaled orthogonal factor 并保持 Frobenius norm。

**数值边界必须保留在交接中：** 当前实现没有在最后再次按输出实测范数精确归一化。特征值 clipping、秩亏和低精度会导致范数偏差或不严格正交；秩亏时 polar 乘法不会凭空补齐缺失方向。这是现有实现行为，后续若修复需要新版本记录。

### 5.3 每一步的确切顺序

1. 如已到 warmup 阈值，首次建立固定 SRR mask，并安装梯度 hook。
2. task+两项 CMR 前向与反向；单 GPU 累积 16 次 microbatches。
3. 完成梯度裁剪和 `optimizer.step()`。
4. 检查 Accelerate 是否跳过 optimizer step；仅成功 step 调用 `_post_optimizer_step()`。
5. 对所有 226 个 LoRA layer 的 A/B 执行回缩。
6. 对 LLM A/B 重新应用固定 SRR mask。
7. 计算回缩前后范数、mask 前后正交残差及非零比例，写入进度/日志。
8. 成功 step 才推进 scheduler；清梯度并更新 global_step。

回缩与 mask 都在 `torch.no_grad()` 内。当前没有切空间梯度投影，也没有把 Adam 一阶/二阶状态作向量运输。被 mask 的坐标可能在 optimizer 状态里仍有历史动量，但其参数最终被重新置零；回缩也可能在临时满矩阵里产生这些坐标。

### 5.4 为什么 SRR 与流形有冲突

稀疏约束集合与正交约束集合通常不一致。一般有

\[
\mathcal M\odot\operatorname{Retr}(X)
\ne\operatorname{Retr}(\mathcal M\odot X).
\]

先回缩后 mask 保住固定 support，但破坏精确正交；先 mask 后回缩可能重新产生 mask 外非零，破坏 SRR。不能简单调整顺序就声称同时严格满足两者。

当前选择前者，优先保住原 SRR 行为，并报告最终实际残差。它应理解为**带固定稀疏掩码的欧氏 AdamW 后处理回缩**。

### 5.5 一个重要理论观察：可能过度均衡奇异值

在没有 mask 且两个因子都满足理想 scaled Stiefel 条件时，可写

\[
B=s_BQ_B,\qquad A=s_AQ_A^\top,
\]

其中 Q_B、Q_A 列正交。于是

\[
\Delta W=\frac{\alpha}{r}s_Bs_AQ_BQ_A^\top.
\]

其 r 个非零奇异值相等。当前 mask 会打破这项理想性质，但这个观察揭示了方法不仅“去方向冗余”，还可能压平更新谱。预训练适配任务不一定需要所有方向同等强度，这为后续流形方向创新提供了具体入口。

## 6. 已完成实验设置

| 参数 | 已完成流形实验实际值 |
|---|---|
| Backbone | LLaVA-v1.5-7B |
| Vision tower | CLIP ViT-L/14-336，本地 `clip-vit-large-patch14-336` |
| Connector | `mlp2x_gelu`，两个线性层均有 LoRA |
| 训练集 | 固定 IconQA 10,000 条，不重新抽样 |
| 数据文件 | `assets/data/iconqa/iconqa_txt-train-10000k-seed0.json` |
| 图像目录 | `assets/data/iconqa/raw/data/iconqa_data` |
| LLM ranks | 16、32、64 |
| Connector rank | 128 |
| 两类 alpha | 64 |
| LoRA dropout / bias | 0.05 / none |
| Epochs / optimizer steps | 3 / 1875 |
| 单 GPU batch / accumulation | 1 / 16 |
| Global batch | 16 |
| LLM / connector LR | 2e-4 / 2e-5 |
| Optimizer | `torch.optim.AdamW`，betas=(0.9,0.999)，eps=1e-8 |
| Weight decay | 0 |
| Scheduler / warmup | cosine / ratio=0.03 |
| SRR warmup / keep | 100 optimizer steps / 每个 LLM 因子 10% |
| CMR 系数 | LLM=1e-3，connector=1e-5 |
| 原正交项 | 关闭，ORTHO_LAMBDA=0 |
| 精度 | bf16=True、tf32=True |
| Gradient checkpointing | True |
| DeepSpeed | 正式流形训练未启用，但代码 import 仍依赖安装包 |
| Model max length | 2048 |
| Image aspect ratio | pad |
| group_by_modality_length | True |
| Seed / data_seed | 42 / 42 |
| dataloader workers | 4 |
| Prompt / decoding | vicuna_v1 / temperature=0，确定性生成 |
| GPU | NVIDIA A100 80GB PCIe，每个 rank 单 GPU |

1875 的来源：10000/16=625 optimizer steps/epoch，3×625=1875。microbatch step 不等于 optimizer step；回缩和 SRR warmup 都按 optimizer step 计。

文件名中的 `10000k` 是历史名称，实际 JSON 列表经本次核对恰好 10000 条；`seed0` 是固定数据子集的抽样标记，不是正式训练 seed=0。

manifest 最终记录的历史 allocation 为 `8378610`，节点 `gpu-a100-02.deac.wfu.edu`。不要根据这个记录推断当前卡仍被保留。

## 7. 数据和评测协议：迁移时必须原样保留

| Benchmark | Questions | Images | 本次实际样本数 |
|---|---|---|---:|
| OKVQA | `assets/data/eval/okvqa/okvqa_val.jsonl` | `assets/data/eval/okvqa/val2014` | 5046 |
| OCRVQA | `assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl` | `assets/data/eval/ocrvqa/sampled_images` | 5000 |
| GQA | `assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl` | `assets/data/eval/gqa/data/images` | 12578 |
| TextVQA | `assets/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl` | `assets/data/eval/textvqa/train_images` | 5000 |

评分额外依赖 OKVQA 的原 questions/annotations、GQA `data/testdev_balanced_questions.json`、TextVQA `TextVQA_0.5.1_val.json` 等。建议迁移完整四个 benchmark 子目录，不只搬 JSONL。

OCRVQA 是已有 deterministic 5000-example subset，不允许重新采样、换一批图或静默替换为完整测试集。GQA 已完成结果采用 testdev-balanced **accuracy fallback**，因为完整 scene-graph evaluator 所需文件当时不可用。TextVQA 问题文件含 OCR 信息，不能随意更换 prompt。

GQA 脚本会根据 scene graph/choices/evaluator 文件是否存在自动选择分支：新服务器如果多出这些文件，指标路径可能改变。复测历史结果前必须确认仍使用相同 fallback；若采用官方 evaluator，应新建协议标签并重新计算可比 baseline。

不应把“本地使用了相同模型和超参数”写成“与论文官方协议完全一致”。明确已知差异是 OCRVQA 子集和 GQA 评分路径；其他一致性仍要由论文、原作者代码和样本/配置证据共同核验。

### 7.1 已核对的数据指纹（SHA-256）

```text
IconQA 10000:
89ba123884d07eef2c977d4c4b27a053e4f2f0b91c1a69c15c355919f2599fe1
OKVQA 5046:
daa1e91a0934a0f7ef4f6aa5790c93a26b8868e4eead987e1012abed9def53e7
OCRVQA 5000:
6d6c961636a070fe59ebb1c2daf8849c1c7d7ea8c31d51e3c8c10f8b5af600a9
GQA 12578:
515aab3142889269480c71212d3b5b8118704d8da6637d767a0cb5bdf80a3d35
TextVQA 5000:
9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3
```

这些只校验列出的 JSON/JSONL 字节，不代表模型、图像、原始 annotations 已完整校验。迁移应另外生成这些资产的文件清单和 checksum。

## 8. 正式结果与对照

分数单位为百分比；差值单位为百分点。Source Avg 是四个百分比分数的等权算术平均，不是把四个数据集样本混在一起计算 accuracy。

\[
\mathrm{SourceAvg}=\frac{\mathrm{OKVQA}+\mathrm{OCRVQA}+\mathrm{GQA}+\mathrm{TextVQA}}4.
\]

### 8.1 三类数据来源并列

| 来源 | Rank | OKVQA | OCRVQA | GQA | TextVQA | Source Avg |
|---|---:|---:|---:|---:|---:|---:|
| TPAMI 论文参考值 | 16 | 57.34 | 62.10 | 60.96 | 57.40 | 59.4500 |
| 已有本地 LoRASculpt+ | 16 | 57.40 | 63.42 | 61.22 | 57.79 | 59.9575 |
| masked AB-manifold | 16 | 57.92 | 63.50 | 61.79 | 57.99 | 60.3000 |
| TPAMI 论文参考值 | 32 | 57.23 | 64.50 | 60.86 | 57.38 | 59.9925 |
| 已有本地 LoRASculpt+ | 32 | 57.30 | 62.98 | 61.38 | 58.05 | 59.9275 |
| masked AB-manifold | 32 | 58.00 | 63.70 | 61.78 | 57.97 | 60.3625 |
| TPAMI 论文参考值 | 64 | 56.65 | 63.90 | 59.84 | 55.48 | 58.9675 |
| 已有本地 LoRASculpt+ | 64 | 57.46 | 62.74 | 61.81 | 57.82 | 59.9575 |
| masked AB-manifold | 64 | 58.20 | 63.64 | 62.12 | 58.08 | 60.5100 |

论文参考值来自现有 manifest/reference 表及用户指定对照。本次未重新解析论文表格，也未核实论文 seed 数量，因此论文表号、原始 seed 数和是否多次平均必须在正式写作前另行确认。

### 8.2 相对本地 baseline

| Rank | ΔOKVQA | ΔOCRVQA | ΔGQA | ΔTextVQA | ΔSource Avg |
|---|---:|---:|---:|---:|---:|
| 16 | +0.52 | +0.08 | +0.57 | +0.20 | +0.3425 |
| 32 | +0.70 | +0.72 | +0.40 | -0.08 | +0.4350 |
| 64 | +0.74 | +0.90 | +0.31 | +0.26 | +0.5525 |

### 8.3 相对论文报告值

| Rank | ΔOKVQA | ΔOCRVQA | ΔGQA | ΔTextVQA | ΔSource Avg |
|---|---:|---:|---:|---:|---:|
| 16 | +0.58 | +1.40 | +0.83 | +0.59 | +0.8500 |
| 32 | +0.77 | -0.80 | +0.92 | +0.59 | +0.3700 |
| 64 | +1.55 | -0.26 | +2.28 | +2.60 | +1.5425 |

当前可以准确写：“在本地固定评测协议及单 seed 设置下，masked AB-manifold 的三个 rank 四项等权平均均高于已有本地 baseline。”

不能写：“所有 benchmark 都提升”“已严格超过论文”“已证明减少灾难性遗忘”。rank32 的 TextVQA 低于本地 baseline，rank32/64 的 OCRVQA 低于论文参考值。上游表现更好与知识保留假设相容，但要证明减少遗忘，还需 pretrained 未微调起点、适配任务效用和多 seed 证据。

## 9. 时间、显存及实际诊断

### 9.1 已完成运行成本

| Rank | Train seconds | Train hours | Eval seconds | Eval hours | 记录 Peak VRAM |
|---|---:|---:|---:|---:|---:|
| 16 | 18918 | 5.25 | 3264 | 0.91 | 33743 MiB |
| 32 | 20429 | 5.67 | 3304 | 0.92 | 33743 MiB |
| 64 | 24264 | 6.74 | 3304 | 0.92 | 33743 MiB |

以上为历史 A100 80GB PCIe 的一次运行，不能保证新服务器同速。每个 rank 训练+评测约 6.2、6.6、7.7 小时；串行合计约 20.4 小时，不含数据迁移、环境编译、排队及异常恢复。三张等效卡并行通常受 rank64 总耗时约束，但实际调度和 I/O 可能增加时间。

**显存统计限制：** 当前监控脚本每 10 秒通过 `nvidia-smi --query-gpu` 采集可见设备，记录函数取 CSV 所有行的最大 memory.used。三组相同 33743 MiB 可能含节点其他 GPU 的占用；不能当作经严格进程归属验证的每个 rank 精确峰值。它也不是 `torch.cuda.max_memory_allocated()`。后续需要同时保存 allocated/reserved 峰值与指定 GPU UUID 的进程采样。

### 9.2 时间戳（统一 UTC）

| Rank | Train start | Train end | Eval start | Eval end |
|---|---|---|---|---|
| 16 | 2026-10-07 09:13:35 | 2026-10-07 14:28:53 | 2026-10-08 05:12:51 | 2026-10-08 06:07:15 |
| 32 | 2026-10-07 09:13:35 | 2026-10-07 14:54:04 | 2026-10-08 05:12:51 | 2026-10-08 06:07:55 |
| 64 | 2026-10-07 09:13:35 | 2026-10-07 15:57:59 | 2026-10-08 05:12:51 | 2026-10-08 06:07:55 |

时间来自 progress 中记录的阶段 wrapper 时间；individual evaluation_metadata 的 start 可能晚十几秒。历史训练完成到评测之间存在数据/加载修复等待，因此“训练完立刻评测”不是这批历史记录的实际情况。

### 9.3 最后一次记录的 AB diagnostics

| Rank | A mask 后残差（最大） | B mask 后残差（最大） | A 范数前→回缩后（均值） | B 范数前→回缩后（均值） | A/B 非零率（层均值） |
|---|---:|---:|---|---|---|
| 16 | 0.0030816684 | 0.0002719985 | 1.5144863→1.5144720 | 0.6081476→0.6081413 | 0.1079562 / 0.1079568 |
| 32 | 0.0030761140 | 0.0002354011 | 2.1341190→2.1341259 | 0.8225835→0.8225808 | 0.1079631 / 0.1079630 |
| 64 | 0.0031618243 | 0.0003795793 | 3.0247155→3.0247154 | 1.1021092→1.1021106 | 0.1079633 / 0.1079634 |

残差定义：

\[
E_B=\left\|B^\top B-\frac{\|B\|_F^2}{r}I_r\right\|_F,\qquad
E_A=\left\|AA^\top-\frac{\|A\|_F^2}{r}I_r\right\|_F.
\]

当前诊断统计 226 个层：范数为层均值，残差为层最大值，非零率为各层比例的简单均值。非零率约 10.796% 不是 SRR 保留率错成 10.8%：224 个 LLM layer 接近 10%，两层 connector 未 mask 通常接近 100%，混合平均约 `(224×0.1+2×1)/226=0.107965`。

当前记录的是数值非零率，并非显式 bool mask 的保留率，两者应区分。后续必须分别统计 LLM 固定 mask 比例、LLM 实际非零率和 connector 非零率。

这些 diagnostics 是 progress 中最后一次采样，不保证精确对应最终 step=1875，因为写入有 10 秒节流。应在正式写论文前直接加载最终 checkpoint 重新逐层统计。绝对残差依赖因子尺度，后续需再报归一化残差，例如 `||G-cI||_F / (||cI||_F+epsilon)`。

## 10. Checkpoint、结果和历史失败的保存位置

三个正式 checkpoint：

```text
checkpoints/ABManifold-r16/
checkpoints/ABManifold-r32/
checkpoints/ABManifold-r64/
```

每个目录包含 `adapter_model.bin`、`adapter_config.json`、`config.json`、`lorasculpt_plus_config.json`、`non_lora_trainables.bin`、`trainer_state.json`、`evaluation_metadata.txt` 和 `eval_results/`。adapter 大小约 160 MiB、312 MiB、617 MiB；它们不是完整 7B backbone，推理还需要原 backbone 与 CLIP。

每项结果模式：

```text
checkpoints/ABManifold-rR/eval_results/DATASET/llava-v1.5-7b/result-DATASET.txt
checkpoints/ABManifold-rR/eval_results/DATASET/llava-v1.5-7b/llava-v1.5-7b-DATASET.jsonl
```

R=16/32/64，DATASET=`okvqa/ocrvqa/gqa/textvqa`。GQA 还保存 `testdev_balanced_predictions.json`。本次核对每个 rank 的合并预测行数均为 5046、5000、12578、5000，结果文件显示分数与上述表格一致。

训练及评测日志分别为 `logs/ABManifold-rR.out`、`logs/ABManifold-rR-eval.out`；GPU 采样分别为 `.gpu.csv` 与 `-eval.gpu.csv`。

### 10.1 历史失败不能删除或误读

日志是 append 模式，因此最终成功日志前可能有早期失败。已读到的失败包括：

- OCRVQA 所需 5000 张图片在预期目录缺失，数据完整性检查中止。
- 非标准 checkpoint 名 `ABManifold-r64` 未正确识别成 LLaVA adapter，导致走普通 AutoModel 加载路径，出现 `Unrecognized configuration class ... LlavaConfig`。
- 视觉塔加载/模型类型路径错误导致 image processor 为 None，报 `NoneType ... image_mean`。

最终代码中的 builder 已包含基于 `adapter_model.bin`/`lorasculpt_plus_config.json` 的 adapter 类型识别与显式本地 LlavaConfig 加载；最终四项预测及评分文件存在。迁移必须保留这些修复。

progress 中 `status=COMPLETED` 可能同时保留早期 `failure_exit_code`/`failure_log` 字段。这是历史状态未清空，不能仅凭它判定最终失败；应结合最后时间戳、trainer state、完整 checkpoint、样本数及最终评分。也不能删除错误记录来让项目“看起来干净”。

## 11. 环境版本与迁移陷阱

实际安装版本经 package metadata 核对：

| Package | Version |
|---|---|
| Python | 项目环境为 Python 3.10 路径；精确 patch 版本迁移时另导出 |
| torch | 2.1.2+cu118 |
| torchvision | 0.16.2+cu118 |
| transformers | 4.37.2 |
| peft | 0.4.0 |
| accelerate | 0.21.0 |
| deepspeed | 0.12.6 |
| numpy | 1.26.4 |
| flash-attn | 2.5.8 |
| bitsandbytes | 0.41.0 |

`pyproject.toml` 不是完整 lock 文件，部分包未严格 pin。迁移前另导出 `pip freeze`、Python 版本、CUDA driver 和 GPU 信息。不要默认升级 PEFT/Transformers 后仍兼容自定义 trainer 与旧 LoRA 类导入路径。

当前重要软链接：

```text
LoRASculptPlus_OptimizerAblations/assets -> ../LoRASculpt/assets
LoRASculptPlus_OptimizerAblations/.conda -> ../LoRASculpt/.conda
```

只复制 OptimizerAblations 目录而保持这两条链接，新服务器没有相应 sibling `LoRASculpt` 时就会缺模型、数据、Python。资产内部也可能存在软链接，必须清点并处理。

不要盲目 `rsync -L` 整个实验根目录：它会把环境和大量共享数据一起展开，容量可能远超预期。先清点真实大小与链接目标，选择完整 sibling 布局，或只解引用所需 assets 子目录并在新服务器重建环境。本文没有执行任何复制或清理。

`setup_env.sh` 设置 LORASCULPT_ROOT、PYTHONPATH、缓存目录并尝试加载集群 CUDA module。它默认把 `.conda/bin` 放入 PATH；新服务器应改为实际新环境路径。迁移后还需把 TMPDIR、缓存目录改为新用户可写位置。

checkpoint 的 `config.json` 中 `mm_vision_tower` 等字段可能保留原绝对路径。训练 JSON 如出现绝对 image path、adapter config 的 base_model_name_or_path 也需审查。只改 shell 中 ROOT 不一定够；修改必须在副本中进行并记录旧值与新值。

### 11.1 本次轻量检查的结果

整理文档时在原服务器只运行了两个 CPU 数学单元测试：范数/正交回缩测试、零初始化 B 跳过测试，均 PASS，没有执行训练。

直接调用原环境 Python 时还出现 bitsandbytes 无 GPU 编译支持提示，以及 DeepSpeed 退出时尝试写只读 `/home/qianr/.triton` 的缓存异常。二者不应被隐瞒，也不能据此否定历史已完成 GPU 训练。新环境应设置可写 `TRITON_CACHE_DIR`，实际 GPU smoke 必须重新确认；当前流形方法使用普通 AdamW，不依赖 bitsandbytes 8-bit optimizer。

## 12. 推荐迁移内容清单

### 12.1 必须迁移

- 全部项目源代码、pyproject、setup/env 脚本和四项 eval 脚本，尤其保留基础 trainer 和 builder 修复。
- 三个 ABManifold 正式 checkpoint 及全部四项预测/评分、metadata。
- ABManifold 训练/评测日志、GPU CSV、progress 单 rank JSON。
- `results.md`、`results.csv`、`run_manifest.json`，以及本地 baseline 表和本文。共享汇总包含别的历史行，保留溯源即可，后续不启动它们。
- 本地 LLaVA-v1.5-7B、CLIP ViT-L/14-336、完整 tokenizer/model config。
- 固定 IconQA 10000 JSON 和其引用的训练图片。
- 四项 benchmark 的 questions、annotations、图像及实际 fallback scorer。
- 环境清单、代码 commit/文件 hash、数据文件 hash；没有 commit 可用时按文件生成 checksum。

### 12.2 可以按研究需要选择

原始下载压缩包、未用于本次的完整数据集、其他 benchmark、缓存目录、历史 allocation 日志可选择归档。不应因“节省空间”删除原服务器唯一数据；检查迁移校验成功前保留源文件。

### 12.3 新服务器验收顺序

1. 检查磁盘可用容量、soft links、文件数量和 checksum。
2. 检查环境版本，CUDA 可见性及单 GPU bf16 支持。
3. 检查五个训练/问题文件的 hash 和样本数。
4. 运行数据完整性检查，确认图片存在。
5. 加载 mixed-rank adapter，确认 224+2 层、rank/alpha、vision processor 和图像输入。
6. 只跑流形单元测试，再做 2 optimizer steps 的真实训练 smoke。
7. 如需验证迁移后推理一致性，先在独立临时输出目录复测已有 checkpoint；不能覆盖历史 predictions。
8. 以上通过后再启动新研究版本或多 seed 实验。

## 13. 新服务器上的可执行操作模板

以下假设项目最终放到 `/home/weitaoma/LoRASculptPlus_OptimizerAblations`；这只是模板，实际落盘目录以接收服务器确认结果为准。不要从本文推断目标服务器已经有这些文件。

### 13.1 环境与数据检查

```bash
cd /home/weitaoma/LoRASculptPlus_OptimizerAblations
# 先激活在新服务器重建的 Python 3.10 环境。
export LORASCULPT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export HF_HOME="$PWD/.cache/huggingface"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$TMPDIR"
nvidia-smi
python -c 'import torch,peft,transformers; print(torch.__version__,peft.__version__,transformers.__version__); print(torch.cuda.is_available(),torch.cuda.device_count())'
python scripts/verify_upstream_eval.py --eval-root "$PWD/assets/data/eval"
sha256sum assets/data/iconqa/iconqa_txt-train-10000k-seed0.json \
  assets/data/eval/okvqa/okvqa_val.jsonl \
  assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl \
  assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl \
  assets/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl
```

### 13.2 只运行流形单元测试

```bash
python -m py_compile llava/model/ab_manifold.py \
  llava/train/LoRASculptPlusOptimizerTrainer.py
PYTHONPATH="$PWD:$PWD/tests" python - <<'PY'
from test_optimizer_ablations import (
    test_scaled_retraction_preserves_norm_and_makes_factors_orthogonal,
    test_zero_initialized_B_is_left_unchanged,
)
test_scaled_retraction_preserves_norm_and_makes_factors_orthogonal()
test_zero_initialized_B_is_left_unchanged()
print("AB_MANIFOLD_UNIT_TESTS=PASS")
PY
```

文件导入可能加载同目录其他 optimizer 的类，这是共享代码依赖，不代表执行对应实验。不要直接运行全套“all experiments”脚本来做流形验收。

### 13.3 新目录内 2-step GPU smoke

在新服务器副本内使用新输出名，避开已有 smoke/checkpoint。外部先分配单 GPU；在 Slurm 下沿用调度器分配的 CUDA_VISIBLE_DEVICES，不硬编码物理卡。

```bash
export OPTAB_METHOD=ab_manifold
export OPTAB_METHOD_LABEL=ABManifold
export OPTAB_RANK=32
export OPTAB_GPU="${CUDA_VISIBLE_DEVICES:-assigned-single-gpu}"
export OPTAB_OUTPUT="$PWD/checkpoints/ABManifold-migration-smoke-r32"
export OPTAB_PROGRESS_FILE="$PWD/progress/migration-smoke-r32.json"
export OPTAB_LOG="$PWD/logs/migration-smoke-r32.out"
export OPTAB_CHECKPOINT="$OPTAB_OUTPUT"
export OPTAB_MAX_STEPS=2
export ORTHO_LAMBDA=0
export CMR_LLM_LAMBDA=1e-3 CMR_CONNECTOR_LAMBDA=1e-5
export STEP_THRESHOLD=100 AB_PRESERVE_RATIO=0.1 OMEGA=1.0
mkdir -p "$PWD/logs"
bash scripts/train_optimizer_config.sh > "$OPTAB_LOG" 2>&1
```

检查 loss finite、参数实际变化、日志中 actual optimizer=AdamW、回缩统计出现、scheduler 推进、adapter/config 可保存并正确加载。两步 smoke 不足以验证第 100 步后的 SRR：额外做一个明确标为测试用途的短 warmup 测试，或跑到 >100 steps；测试改阈值不允许带入正式配置。

### 13.4 正式复跑模板：避免覆盖已有实验

不要直接用 `run_one_optimizer_experiment.sh` 对已有 r16/r32/r64 再跑，因为它固定输出到原名，且训练配置使用 overwrite_output_dir。要复跑时使用新目录/新标签并明确未存在输出：

```bash
unset OPTAB_MAX_STEPS
export OPTAB_METHOD=ab_manifold
export OPTAB_METHOD_LABEL=ABManifold
export OPTAB_RANK=16  # 分别使用 16、32、64
export OPTAB_OUTPUT="$PWD/checkpoints/ABManifold-recheck-seed42-r$OPTAB_RANK"
export OPTAB_PROGRESS_FILE="$PWD/progress/recheck-seed42-r$OPTAB_RANK.json"
export OPTAB_CHECKPOINT="$OPTAB_OUTPUT"
export OPTAB_LOG="$PWD/logs/recheck-seed42-r$OPTAB_RANK.out"
test ! -e "$OPTAB_OUTPUT" || { echo 'Output exists; stop to avoid overwrite'; exit 1; }
bash scripts/train_optimizer_config.sh > "$OPTAB_LOG" 2>&1
```

这个训练脚本当前固定 seed=42、data_seed=42。多 seed 正式实验需在**新的代码分支/副本**把 seed 参数化，并记录实际 seed；不要只换输出文件名而仍使用 42。

一张 GPU 优先串行训练和评测；两张或三张 GPU 可为独立 rank 分别启动单 GPU 进程。每个进程 global batch 始终 16。不要仅为利用满显存就在同卡运行多个 7B 训练：先测吞吐、显存峰值和 I/O，避免 OOM 或总时间反而变长。

### 13.5 已有 checkpoint 的四项评测模板

使用独立输出目录，保留历史结果。若代码已经迁移到新 root，下面路径自动跟随；仍要检查 config 内 CLIP 绝对路径。

```bash
RANK_TO_EVAL=64
MANIFOLD_CKPT="$PWD/checkpoints/ABManifold-r$RANK_TO_EVAL"
MANIFOLD_EVAL_OUT="$PWD/migration_eval/ABManifold-r$RANK_TO_EVAL"
MANIFOLD_EVAL_META="$MANIFOLD_EVAL_OUT/evaluation_metadata.txt"
mkdir -p "$MANIFOLD_EVAL_OUT"
bash scripts/v1_5/eval/upstream/eval_okvqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
bash scripts/v1_5/eval/upstream/eval_ocrvqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
bash scripts/v1_5/eval/upstream/eval_gqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
bash scripts/v1_5/eval/upstream/eval_textvqa.sh "$MANIFOLD_CKPT" "$MANIFOLD_EVAL_OUT" "$MANIFOLD_EVAL_META"
```

评测脚本在 CUDA_VISIBLE_DEVICES 未设置时可能默认 `0,1,2,3`。必须在外部保证每个评测进程只看到自己分配的一张 GPU；不要在共享服务器无 allocation 时直接复制这些命令运行。

### 13.6 Slurm 与查看 GPU

已有有效 allocation 时，示意单 GPU job step：

```bash
squeue -j <实际allocation_job_id>
scontrol show job <实际allocation_job_id>
srun --jobid=<实际allocation_job_id> --exclusive --ntasks=1 \
  --gres=gpu:1 bash <新服务器自己的单rank脚本>
```

实际 GPU type、CPU、内存和分区参数按新集群配置调整。这里不是旧 job 8378610 的启动指令，不保证新服务器使用 Slurm。

进入已分配计算节点观察，需使用一个不抢占训练 step 资源的交互 step；具体权限/资源参数由集群策略决定，例如：

```bash
srun --jobid=<正在RUNNING的allocation_job_id> --overlap --ntasks=1 --pty bash -l
hostname
nvtop
# 没有 nvtop 时：
watch -n 2 nvidia-smi
```

Pending allocation 不能进入；Expired/Invalid job ID 不能复用。nvtop 看到的任务需结合用户名、PID、命令及 allocation 判断是否为自己的实验，不只看显存占用。

## 14. 中断、保存与进度维护

`adapter_model.bin` 是推理/适配权重，不是完整可精确续训 checkpoint。精确续训还需要 optimizer、scheduler、RNG、trainer state、固定 SRR mask，并核对 A/B post-step 行为。

当前正式脚本 `save_steps=10000`，而全部训练只有 1875 steps，所以没有常规周期 optimizer checkpoint。最终 rank64 文件清单未见 optimizer.pt、scheduler.pt 或 SRR mask 文件。迁移后可以加载它评测，但不能直接称为“从最终 checkpoint 精确恢复训练状态”。

训练入口会检测输出目录中的 `checkpoint-*` 决定是否自动 resume；若后续要有可靠恢复，应在新版本配置中降低 save_steps（例如 250），验证 masks/RNG/optimizer/scheduler 都落盘并做断点一致性测试。保存频率变化要在 manifest 记录，不改变模型、数据和学习率日程。

进度建议按每个 run 单独 JSON，再原子生成总表。必须保留 rank、seed、GPU UUID、current_step/1875、epoch、loss、LLM/connector LR、optimizer、残差、mask 保留率、开始结束时间和日志路径。

推荐状态流：`WAITING → SMOKE_RUNNING → TRAINING → TRAINED → EVALUATING → COMPLETED`，异常为 `FAILED`。成功训练但缺某 benchmark 应标记 incomplete，不填 0，不复制其他 rank 结果。

历史 `progress/progress.md` 汇总包含其他方法。新研究进度表只列流形分支；不要通过改旧表或删其他行抹去历史结果。

## 15. 为什么可能提升，以及还缺什么证据

合理解释有三个：减少方向冗余，提高固定 rank 下的有效方向利用；通过每步几何整理减少训练中因子结构漂移；替代一个需要系数平衡的软正交项，同时保留 CMR/SRR 的知识保留机制。

这些是待检验机制假设。当前成绩本身不能证明原因。还存在替代解释：回缩额外引入了因子谱均衡、改变了 AdamW 的有效更新幅度、与固定 mask 共同产生新的正则化，或者 seed 波动。

先做以下对照才能把故事讲清楚：

- 本地 LoRASculpt+：原正交 loss、无回缩。
- 去掉正交 loss、无回缩：隔离“去掉 loss”的作用。
- 去掉正交 loss、加入当前回缩：现有方案。
- 仅 A、仅 B、A+B 回缩：区分两因子的贡献。
- 仅 LLM、仅 connector、两者回缩：定位改进来源。
- 保留/去掉 SRR，或不同 support 处理：检验正交与稀疏冲突。
- 仅保持范数而不强制正交，以及可学习谱版本：检验提升是否由尺度/谱变化驱动。

以上属于后续新研究实验，不代表本次已运行。已有 baseline 不必为交接重训，但要做多 seed、官方完整协议或新代码公平消融，往往需要在匹配设置下补充 baseline。

## 16. 建议主攻路线：让稀疏与几何约束相容

推荐优先把现有“dense 回缩后硬 mask”升级为**固定 support 内的几何优化**。这是当前代码直接暴露出的具体问题，比只把普通 QR/polar 应用于 LoRA 更有明确研究动机。

可在固定 SRR support 上定义

\[
\mathcal F=\{(A,B):A=\mathcal M_A\odot A,\ B=\mathcal M_B\odot B\},
\]

研究在 \(\mathcal F\) 内降低 Gram residual 的更新，而不是每次生成 dense 矩阵再丢掉部分方向。必须先分析固定任意元素 support 是否存在目标满秩正交解；不能默认交集非空或光滑。

分两阶段实现更稳妥：第一阶段保留 mask 和任务梯度，在 support 内作若干步带范数控制的 Gram 修正并报告收敛/耗时；第二阶段若理论允许，构造 support-aware tangent projection/retraction。方法可以允许近似正交，但约束误差与额外计算成本应明确。

这一研究路线的待验证优势是：减少“回缩恢复 mask 外元素、再被清零”的无效计算；在知识保留 support 内改善方向多样性；更直接解释与 SRR 的协同。是否真的更好需实验，不应提前命名成严格稀疏 Stiefel 优化器。

## 17. 第二条流形路线：方向与幅度分离

针对第 5.5 节理想均衡奇异值的问题，考虑

\[
\Delta W=Q_B S Q_A^\top,\qquad
Q_B^\top Q_B=I,\quad Q_A^\top Q_A=I,
\]

其中 S 是可学习对角尺度或小 core。Q 保证方向多样性，S 允许重要方向更强、次要方向更弱，避免所有非零奇异值被同等约束。

这是未来方案，尚未实现。若增加 S 参数，必须报告参数开销并与相同参数预算的对照比较；LoRA alpha/r scaling、初始化、checkpoint 格式和 CMR 作用对象都要重新定义。不要直接把 dense S 与当前逐元素 SRR 拼接后声称保持原 SRR 等价。

可从 diagonal S 开始，优先验证是否改善 rank16 的容量，再扩展 rank32/64。核心消融是：当前 scaled polar、保留可学习方向尺度的版本、仅谱重缩放、不带正交的普通 LoRA。需要同时报告更新奇异值、effective rank、任务表现和上游保留。

## 18. 第三条流形路线：真正的切空间更新

对列正交 Q，Euclidean 梯度 G 的常见 Stiefel 切空间投影为

\[
\operatorname{Proj}_Q(G)=G-Q\operatorname{sym}(Q^\top G),\qquad
\operatorname{sym}(H)=\frac{H+H^\top}{2}.
\]

随后在切空间更新并回缩，而不是每步先做普通 AdamW 再投影。若采用带动量的 Riemannian optimizer，还需处理状态的切空间一致性/向量运输；当前代码没有这些步骤。

这可以作为第二阶段技术升级，但“使用现成流形优化器”本身不必然构成足够新颖的 CVPR 方法。更有说服力的是把优化几何与多模态知识保留、稀疏约束或方向幅度分离联系起来，并给出数学问题、算法和可复现证据。

## 19. 发表导向的实验与证据计划

推荐先 rank16 快速开发/淘汰，再用三个 rank 验证最终方法。最终主表必须在匹配协议下至少多个 seed（例如 42/43/44），报告 mean±std 和逐 benchmark 差值；原论文是否多 seed 本次未核实，不能编造。

除了 accuracy 还应保存：task loss 和 CMR 分量、回缩前/后/mask 后归一化 Gram residual、A/B 范数、各层非零率、实际 \(\Delta W\) 的范数和谱摘要、单步耗时、峰值显存、最终 checkpoint 校验。

样本级 bootstrap 可以估计评测样本不确定性，但不能替代训练 seed 方差。不要用四个 benchmark 当作四次独立训练来做统计显著性。

如果科研范围进一步需要适配任务表现，应单独约定合法 held-out 评测、与上游主表分开呈现；当前用户明确要求不执行 IconQA evaluation，不能以“完善论文”为由擅自运行。

正式写作前补齐：官方协议核验、固定完整样本评测或明确本地子集协议、匹配 baseline、多 seed、机制消融、效率、至少一种额外 backbone 或任务泛化（需另行计划），以及相关流形 LoRA/正交适配文献的新颖性检查。

当前最稳妥的项目叙述是：“现有 masked AB-manifold 提供了稳定方向的初步实验信号，下一步解决稀疏 support 与几何约束的冲突，并检验方向/幅度分离是否进一步提高知识保留与适配效率。”这是一条研究路线，不是已证明的结论。

## 20. 接收 agent 的执行约定（可直接交给另一台服务器）

你接手的是 LoRASculpt+ 的流形优化改进项目。先阅读本文并核对收到的代码/数据/结果，不要只凭对话历史猜测状态。已完成 masked AB-manifold r16/r32/r64 的训练与 OKVQA/OCRVQA/GQA/TextVQA 评测，已有 checkpoint 和结果必须只读保留。

先处理软链接、重建 Python/CUDA 环境、修复副本绝对路径、验证数据 hash、图像、mixed-rank 加载，再运行流形单元测试和独立 2-step smoke。不要运行 IconQA evaluation，不要启动其他方法的批处理，不要默认旧 Slurm allocation 有效。

如获授权继续研究，使用新的版本目录，保持 backbone、固定 IconQA 10000 子集、3 epochs、global batch16、两类 LR、CMR、alpha 和评测协议不变；只修改明确指定的流形机制。先记录现有实现的限制，再实现 support-aware 或方向/幅度分离方案；每项变化单独消融。

每次实验保存真实 seed、参数清单、Git/file hash、优化器类、SRR mask、残差、损失分量、GPU/显存/时间、checkpoint、四项预测及评分、原子进度表。缺失结果标 incomplete，失败保留错误日志，不填零不替换 rank。只有核验完成才写 completed。

首先向研究者报告收到的资产完整性和可运行状态；正式启动新训练前明确本次方法定义、实验标签及资源安排。不得把本地子集/fallback 的结果写成严格官方论文协议，不得把单次平均提高写成统计显著或可直接发表。

## 21. 最终交接验收清单

- [ ] 已收到本文和原结果/manifest，而不是只收到单个脚本。
- [ ] `assets` 与 Python 环境链接已在新服务器解析。
- [ ] Backbone 与 CLIP 可从本地加载，config 内路径正确。
- [ ] 固定训练集 10000 条，数据指纹匹配。
- [ ] 四项 benchmark 图像/annotations 完整，问题指纹匹配。
- [ ] GQA fallback 与 OCRVQA subset 的协议标签明确。
- [ ] 三个正式 adapter、mixed-rank config 和 trainer_state 可核对。
- [ ] 十二个结果及十二个预测样本数正确。
- [ ] 原始日志与失败记录保留。
- [ ] 两个流形数学测试及新服务器 GPU smoke 通过。
- [ ] 后续输出使用独立名字，原 baseline/流形结果未覆盖。
- [ ] 新版本进度、环境、checksum、seed 和资源归属已记录。

本文完成的是流形方向的项目交接，不包含新训练、迁移成功或新方法提升的虚构声明。
