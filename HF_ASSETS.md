# Hugging Face assets

目标仓库：https://huggingface.co/datasets/weitao040702/LoRASculptPlus_OptimizerAblations

当前状态：2026-10-15，用户授权后已验证仓库公开（private=False）。原私有额度不足的上传进程已停止，单worker后台续传已重启。全部上传尚未完成，以日志中的 `VERIFIED` 为准；公开存储仍受HF平台限制。

上传范围为 `assets/data/`、`assets/models/`、`checkpoints/` 的完整实体内容，包括已有数据包、模型、adapter、训练状态和评测输出。代码、日志、汇总和交接文档保存在 GitHub；Python 环境、缓存及临时文件不上传。

估计数据约140 GB、模型约16 GB、checkpoint约4.4 GB。HF账户仍受存储配额、仓库/文件限制约束；不能保证免费账户可容纳全部内容。保持原数据及模型许可证适用；HF dataset 类型不改变这些文件的许可证。新创建的仓库默认 private；已存在仓库保留原可见性。

## 原服务器认证与上传

该环境中的 `hf` executable 有旧 shebang，因此使用有效 Python 路径：

```bash
cd /deac/csc/yangGrp/qianr/qiushi/mwt/LoRASculptPlus_OptimizerAblations
export HF_HOME="$PWD/.cache/huggingface"
.conda/bin/python -c 'from huggingface_hub import login; login()'
nohup .conda/bin/python -u scripts/upload_hf_assets.py > logs/hf-assets-upload.out 2>&1 < /dev/null &
tail -f logs/hf-assets-upload.out
```

使用 HF write 权限令牌，不使用 GitHub token。不要发送令牌到聊天或提交到Git。脚本先认证，再创建仓库、建立逐文件软链接暂存、调用可恢复大目录上传并核对远程路径/大小；不复制/删除源文件。重跑同一脚本可以利用暂存目录中的上传缓存恢复。上传成功前不要移动原 assets/checkpoints。上传完成以最后 `VERIFIED` 输出为准，不以进程启动为完成。

`asset_inventory.json` 保存计划上传文件的路径和字节大小；此版本验证路径/大小，不声称完成远程内容SHA256独立校验。大文件由HF上传机制校验。

## 接收服务器下载

安装新版 huggingface_hub CLI，并在需要访问私有仓库时登录：

```bash
hf auth login
hf download weitao040702/LoRASculptPlus_OptimizerAblations \
  --repo-type dataset --local-dir /path/to/LoRASculptPlus_OptimizerAblations
```

目录应与GitHub代码目录合并，使 `assets/data`、`assets/models` 和 `checkpoints` 位于项目根目录。检查原模型/adapter配置的绝对CLIP路径并在迁移副本中修改；下载资产不等于Python环境已安装。复现实验细节见 `LoRASculptPlus_MANIFOLD_HANDOFF.md`。
