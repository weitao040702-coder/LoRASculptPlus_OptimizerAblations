# OrbitStiefel v1：新方案实现与实验入口

当前 OrbitStiefel 方法的训练入口为 `scripts/train_new_scheme.sh`；已完成的实验结果见 `README.md`。

| 旧机制 | 新实现 |
|---|---|
| SRR 固定坐标幅值裁剪 | warmup 后 OrbitSparse：平衡、联合正交坐标搜索、按乘积误差选择固定稀疏 support |
| LLM 按预训练权重幅值保护 | 预训练权重近似主左右奇异子空间保护 |
| Connector 加权逐元素 L1 正则 | 每方向可学习尺度的近端稀疏化（本项目定义的 spectral sparse LoRA） |
| A/B scaled polar 后处理 | 稀疏生成因子的 QR Stiefel 参数化 + 每方向可学习尺度 |

ACM 的两侧非对称容量配置仍保留：LLM rank 16/32/64，connector rank 128；学习率分别为 2e-4、2e-5。本版本不声称替换了整个 ACM，也不声称复现了某篇尚未指定的 Sparse LoRA 论文。

## 参数表示与约束

每层使用稀疏生成因子 G_B、G_A，固定正交坐标缓冲 O_B、O_A，和可学习有符号尺度 s：

    Q_B = qr(G_B).Q O_B
    Q_A = qr(G_A^T).Q O_A
    delta = (alpha/r) Q_B diag(s) Q_A^T

QR 使用 FP32，方向保持列正交，允许各方向强度不同。初始尺度为零，初始模型更新为零；第一步尺度先获得梯度，方向随后获得梯度。没有旧正交 loss，没有旧 dense polar 回缩。

**稀疏的是生成因子，不是最终 QR 方向因子或合并后的 delta。**普通稠密 QR/矩阵乘法不承诺吞吐加速。每步计算 QR 有额外成本，需要实际 GPU 测量。这是为解决元素 mask 与严格正交的冲突而选定的具体架构，不是文档 OrbitSparse 原算法的完全相同训练延续方式。

OrbitSparse 在 effective B/A 上形成原始直接剪枝、平衡直接剪枝及旋转候选。剪枝后对稀疏候选进行 QR 和小核心 SVD，将其乘积精确迁入上述参数表示；小核心的左右奇异向量记录为固定坐标缓冲，奇异值初始化 s。这样不错误地假设非均匀尺度与任意旋转可交换。

可微 QR 要求生成因子满列秩，因此候选筛选会拒绝数值秩亏候选。如果直接剪枝候选满秩，则保留其误差回退保证；否则不声称优于被拒绝的候选。若所有候选秩亏，停止并要求调整保留率，不静默改变预算。后续更新仍可能使生成因子接近秩亏，正式训练需关注数值稳定性。

剪枝点仅重置被重参数化的 LLM adapter 参数 Adam 状态，scheduler 不重置；connector 状态保留。掩码、剪枝标记、固定坐标和尺度均进入训练状态。梯度、参数及形状匹配的 Adam 状态在固定 support 外置零。

## 知识保护

仅覆盖原本地实现的 LLM q/k/v targets，避免同时扩张保护范围。冻结预训练权重 W 的 leading U_k、V_k 通过固定随机种子的 randomized SVD（oversampling=8，两个 power passes）近似计算，不宣称精确 top-k SVD。

    L_protect = lambda * sum(||U_k^T delta||_F^2 + ||delta V_k||_F^2)

这是同时限制进入主要输出子空间、以及作用于主要输入子空间的更新；交集会被双重计入。本版本采用软正则，不承诺严格无扰动或不遗忘。计算包含实际 alpha/r，使用低秩收缩避免形成完整 delta。

Connector 原重要性加权矩阵和 L1(BA) 均移除。Connector 的新稀疏目标是实际更新的 nuclear norm，即 (alpha/r) sum |s_i|，由 Adam 对角度量下的 proximal soft-threshold 更新实现，task 梯度之外不重复添加一次 L1 梯度。这不是固定 rank 预算；方向可变为零，也允许后续任务梯度重新激活。

## 运行

激活已验证的 Python 环境、在外部暴露一张已分配 GPU 后：

```bash
CUDA_VISIBLE_DEVICES=<分配的卡> OPTAB_RANK=16 bash scripts/train_new_scheme.sh
```

配置为 `configs/orbit_stiefel.json`。seed 由 OPTAB_SEED 参数化；训练入口记录实际配置。默认 warmup=100，keep=0.1，坐标搜索10次，保护rank=8。rank=8 与两个正则系数是初始实验选择，尚未验证最优。

输出默认 `checkpoints/OrbitStiefel-v1-seed42-r16`，已有输出会拒绝覆盖。续训需显式 `OPTAB_RESUME=1`，目录必须包含匹配配置和完整周期 checkpoint。保存间隔通过 OPTAB_SAVE_STEPS 设置；论文自动队列采用作者脚本的10000步，因此1875步正式训练只有最终保存。支持单设备或普通双卡 DDP、16-bit backbone、bias=none、weight_decay=0；不支持 ZeRO/FSDP/量化训练。双卡的剪枝选择由 rank0 计算并广播，各卡同步重置相同 optimizer 状态，仅 rank0 导出模型和进度。

2-step smoke 可设置 `OPTAB_MAX_STEPS=2 OPTAB_TAG=OrbitStiefel-smoke`，不等于通过第100步后的剪枝验收。需要单独测试 warmup 切换或做 >100步 GPU 验收。正式 seed、数据、batch、LR 仍使用原固定协议。

## 保存与评测

- `orbit_stiefel_state.pt`：训练用生成因子、尺度、坐标、保护子空间、mask和剪枝标记。
- `orbit_stiefel_config.json`：方法配置，续训时严格匹配。
- `adapter_model.bin`：物化为普通 A/B 的推理导出；现有 mixed-rank builder 可加载并合并，不需要特殊推理模块。
- `lorasculpt_plus_config.json`：保留 mixed-rank 布局。
- `orbit_pruning_report.json`：剪枝点候选误差、是否可用、选中候选、状态重置记录。

训练用参数化模型禁止直接调用 PEFT merge（原始生成因子并非 effective A/B）；使用导出的标准adapter进行merge。单独的最终 adapter 仍不等于完整续训状态；续训使用 checkpoint-* 目录的完整状态。没有运行 IconQA evaluation；正式四项评测仍按原协议执行。

## 验证

`tests/test_orbit_stiefel.py` 覆盖数学、梯度、support、近端零尺度、导出与状态恢复；`tests/test_orbit_stiefel_integration.py` 覆盖真实PEFT训练、标准adapter导出/合并、跨剪枝点续训一致性。完成情况以本次实际测试记录为准，不能据此声称已通过7B GPU训练。

## 本次验证环境与结果

在项目内 `.venv_new_scheme` 创建了隔离验证环境，复用服务器现有 PyTorch 2.7.1+cu128。验证依赖为 Python3.10、Transformers4.37.2、PEFT0.4.0、Accelerate0.21.0、NumPy1.26.4。DeepSpeed0.12.6 与此 PyTorch 的 elastic API 不兼容，因此验证环境使用0.16.9，仅作为旧 trainer 的 import 依赖，不启用 DeepSpeed。没有修改共享 conda 环境。

新启动脚本默认使用这个环境；其他环境可用 OPTAB_PYTHON 指定。新方案默认走 `llava.train.train`，不强制旧 FlashAttention2.5.8；attention 内核与正式GPU吞吐需要另行验收。

已验证：五项CPU数学/生命周期测试；真实PEFT四步CPU训练及梯度累积；跨第2步剪枝阈值的checkpoint恢复；标准adapter保存、加载和merge后的输出一致性；真实训练入口 `--help`。另已通过两进程 CPU 和实际两张 GPU 的小模型训练、剪枝状态一致、导出及断点恢复测试，记录见 logs/new-scheme-ddp-gpu-validation-socket.txt。7B smoke 和正式训练由新自动队列等待 IconQA 完整后执行；尚未完成四项正式评测。双卡兼容配置及论文协议范围见 AUTO_PAPER_PIPELINE.md。
