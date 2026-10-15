---
language:
  - en
  - zh
task_categories:
  - visual-question-answering
tags:
  - multimodal
  - lora
  - manifold-optimization
  - research-artifacts
pretty_name: LoRASculptPlus Optimizer Ablations Research Assets
---

# LoRASculptPlus Optimizer Ablations — Research Assets

本仓库保存多模态低秩适配实验的文件资产。它是研究项目资产仓库，**不是一个新标注的数据集，也不是原论文官方仓库**。

代码与详细方法说明：
https://github.com/weitao040702-coder/LoRASculptPlus_OptimizerAblations

## 上传状态 / Upload status

**上传尚未完成，不要将当前快照视为完整备份。** 计划资产清单包含 644,018 个文件，原始文件大小合计 171,417,528,850 bytes（约171.4 GB / 159.6 GiB），不含清单自身。首次上传受私有存储配额限制；2026-10-15 经仓库所有者授权改为公开，并恢复上传。

部分内容可能已传输到后端但尚未提交到仓库。只有仓库文件树中可见的文件才是可下载内容。完成验收需要将远程文件路径和大小与 `asset_inventory.json` 全量匹配；在这之前，本说明不会声称资产完整。

## 目录 / Layout

| 路径 | 内容 |
|---|---|
| `assets/data/iconqa/` | 固定 IconQA 训练子集、图像及本地准备文件 |
| `assets/data/eval/` | 四项上游评测及共享评测目录中的资产 |
| `assets/data/upstream/` | 原服务器保留的上游数据准备／下载资产 |
| `assets/models/llava-v1.5-7b/` | 原本地 LLaVA-v1.5-7B backbone 文件 |
| `assets/models/clip-vit-large-patch14-336/` | CLIP vision encoder 文件 |
| `checkpoints/ABManifold-r16,r32,r64/` | masked AB-manifold 实验的 adapter、配置与评测文件 |
| `checkpoints/Muon-r16,r32,r64/` | 独立 optimizer 消融实验文件 |
| `checkpoints/smoke-*/` | 小规模调试／保存测试产生的文件，非正式结果 |
| `asset_inventory.json` | 计划上传文件的相对路径与字节大小 |

共享 assets 中可能包含额外数据、原始压缩包或重复准备版本。保留它们是为了迁移与溯源，不表示它们都用于正式训练。Python环境、缓存、凭据和临时文件不属于上传范围。

## 实验协议 / Experimental protocol

固定训练数据：`assets/data/iconqa/iconqa_txt-train-10000k-seed0.json`，实际为10,000条样本；IconQA仅用于本项目训练，不作为四项上游结果。

| Benchmark | 本地协议 | 样本数 |
|---|---|---:|
| OKVQA | validation | 5,046 |
| OCRVQA | deterministic固定子集 | 5,000 |
| GQA | testdev-balanced；历史结果使用accuracy fallback | 12,578 |
| TextVQA | validation，含OCR提示 | 5,000 |

本地OCRVQA子集及GQA评分路径与论文官方完整协议可能不同，不能将这些结果直接视为官方协议的严格复现。LoRA LLM ranks为16/32/64，connector rank128；详细超参数、方法及局限见GitHub交接文档。

## 下载与恢复 / Download and restore

```bash
pip install -U huggingface_hub
hf download weitao040702/LoRASculptPlus_OptimizerAblations \
  --repo-type dataset --local-dir /path/to/LoRASculptPlus_OptimizerAblations
```

仓库已公开，通常不需要登录；遇到平台限流可登录本人HF账号后重试。下载目前只获取已经成功提交的文件，不能自动补齐尚未上传的资产。需要较大磁盘空间，训练环境另行安装。

将下载目录与GitHub代码目录合并，保留 `assets/`、`checkpoints/` 相对结构。检查模型/adapter配置中的旧服务器绝对CLIP路径，在迁移副本中修正。adapter不是完整模型，使用时还需backbone与vision encoder。

## 文件格式与验证 / Format and verification

数据包括JSON/JSONL问题与标注、图像、下载压缩包；模型与checkpoint包括PyTorch权重、safetensors及配置。不适合直接假定为单一 `load_dataset()` 表格；请按对应代码读取。

`asset_inventory.json` 的size单位为bytes，只记录计划路径和大小，不是全内容独立SHA256清单。上传完成后仍建议在关键模型/数据文件上做内容校验。仅加载可信来源的PyTorch pickle格式文件。

## 来源、许可证与引用 / Sources, licenses and attribution

这些文件来自多个第三方数据集与模型，本资产仓库**不重新授予统一许可证**。原数据集、模型权重及其上游使用条款仍适用；公开可下载不代表可任意商业使用或重新分发。各文件的完整许可证核验尚未在此汇总；使用者应查阅原始来源。未将未核实的许可证写入HF metadata。

- LoRASculpt / LoRASculpt+：基于原方法的本地复现与研究扩展；原LoRASculpt项目：https://github.com/LiangJian24/LoRASculpt ，论文：https://arxiv.org/abs/2503.16843 。TPAMI扩展版的精确书目信息见项目引用文档，并应在正式发表前核实。
- LLaVA-v1.5-7B：https://huggingface.co/liuhaotian/llava-v1.5-7b
- CLIP ViT-L/14-336：https://huggingface.co/openai/clip-vit-large-patch14-336
- IconQA：https://iconqa.github.io/
- OKVQA：https://okvqa.allenai.org/
- OCR-VQA：https://ocr-vqa.github.io/
- GQA：https://cs.stanford.edu/people/dorarad/gqa/
- TextVQA：https://textvqa.org/

研究使用时请引用各原始模型、数据集及方法。本项目对这些资产的组织与实验修改不构成对原作者成果的所有权声明。

## 限制 / Limitations

用于科研复现与迁移。包含第三方图像、文本和模型可能带有原始偏差或不适当内容。本仓库当前是部分上传状态；不能据此声称已完成完整备份、模型精确恢复或官方协议复现。正式结果、单seed限制、SRR与流形约束冲突、续训状态不足等详见GitHub文档。
