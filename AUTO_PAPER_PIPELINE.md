# 按论文执行的单 seed / 三 rank 实验

依据：本地 LoRASculpt+.pdf 第9页 §4.1、第15页；CVPR LoRASculpt §4.1；作者公开训练脚本和 issue #2 指定的 Mantis 转换脚本（固定 commit 912942133110beaa10dffe0b0ddac0eb851ec61a）。

| 项目 | 执行设置 | 依据 |
| --- | --- | --- |
| 模型 | LLaVA-v1.5-7B + CLIP ViT-L/14-336 | 论文及作者代码 |
| 训练数据 | IconQA choose_txt，随机10000条 | 论文随机10k；作者训练脚本 iconqa_txt |
| 模板 | A. … / B. …；Answer: A | 作者引用的 Mantis prepare.py |
| Seed | 训练42，抽样42 | 用户要求单seed；论文未固定抽样ID，本地记录seed |
| LLM rank | 16 → 32 → 64 | 论文及用户要求 |
| Connector rank | 128 | LoRASculpt+ §4.1 |
| Epoch / global batch | 3 / 16 | 论文 §4.1 |
| 双卡 batch | 每卡4，梯度累积2 | 保持4 × 2 × 2 = 16 |
| 学习率 | LLM 2e-4；connector 2e-5 | 论文 §4.1 |
| 其他超参数 | alpha64、cosine、warmup0.03、weight_decay0、BF16、max_length2048、图像pad | 作者公开训练代码；connector alpha64为记录的本地设置 |
| 训练方法 | OrbitSparse + 主奇异子空间保护 + connector谱尺度稀疏 + 可学习尺度Stiefel | 用户指定新方案，方法专属系数见 configs/orbit_stiefel.json |
| 上游评测 | OKVQA / OCRVQA / GQA / TextVQA；Accuracy | 论文 §4.1 |

新训练文件：assets/data/iconqa/iconqa_txt-train-10000-seed42-paper.json。10000条文本已逐条与作者原始 choose_txt 分支实际执行结果比较；全部图片通过校验。只将 Mantis images/role 数据字段转为 LLaVA image/from，文本值保持一致。原作者训练ID不要求完全相同，也不声称完全相同；当前重建不冒充原作者JSON包。

入口 scripts/auto_paper_pipeline.py。流程：校验训练数据 → 三个rank各跑新模板四步双卡7B短测 → 各rank独立从原始backbone正式训练3epochs/1875steps → 校验评测数据 → 每项两卡分片推理和评分 → 汇总12项正式成绩。短测权重不用于正式训练。旧模板的数据、短测和报告保存在 automation/orbit-paper-seed42/historical-template-*，不会当作新模板通过证据。

GQA使用官方 testdev-balanced 的12578条问题/答案，准确率严格按官方 predicted == gold 计算，预测前处理沿用作者 convert_gqa_for_eval.py。已与未修改的官方 eval.py 在独立参考样例上验证得分一致。sceneGraphs/choices用于官方辅助指标，不是论文Accuracy的必需输入，不再以它们阻塞训练或评分。

OCRVQA已补齐官方测试问答元数据（SHA256校验通过），按作者issue #2的随机2000条协议，使用seed42从split=3的100429条问答中固定抽取2000条，关联1931张图片，全部通过PIL核验。来源、样本ID、抽样顺序、种子和SHA256记录于 configs/paper_eval_provenance.json 及 automation/orbit-paper-seed42/ocrvqa-metadata-provenance.json；可用 scripts/prepare_ocrvqa_paper.py 重建。评分与作者引用的LLaVA-HR一致：去除两端空白、转小写后精确匹配，不使用模糊匹配或额外答案归一化。原作者具体抽样名单未取得，因此这是公开数据上的协议复现，不能宣称与论文OCRVQA结果使用完全相同的2000条；比较新方案与原方案时应共用当前固定子集。

本轮不训练COCO，不运行IconQA Target评测。OKVQA需要COCO val2014图片。汇总只计算四项Source均值；没有Target就不计算论文综合Avg，也不能只据Source断言整体优于论文。

实时表 automation/orbit-paper-seed42/live.md / live.html，每10秒更新。状态 status.json；主日志 logs/auto-paper/pipeline.log；训练日志 r{rank}-train.log。输出 checkpoints/OrbitStiefel-paper-seed42-r{rank}。每个rank保存最终适配器和训练状态；原运行按作者脚本save_steps10000且没有中间checkpoint；2026-10-16恢复修订后，每100步保存完整断点，详见末尾修订记录。

队列使用文件锁和独立后台进程会话。NCCL使用socket通信；只等待两卡显存，不结束其他用户进程。任一步失败会标为FAILED并停队列，数据未齐会明确等待，不生成虚构成绩。


2026-10-16恢复修订：原rank16在1304/1875步因rank1出现SystemError unknown opcode和SIGSEGV退出，报错位于PyTorch非重入梯度检查点pack_hook路径；底层段错误根因未确定，不能声称已彻底修复。失败运行和证据保存在 automation/orbit-paper-seed42/failed-run-20261016-033514。没有中间权重，rank16从原始backbone重跑。新运行关闭gradient checkpointing和周期栈转储、禁用TorchDynamo；训练目标、新方案、seed、batch、epoch和学习率保持。保存间隔由作者脚本10000调整为100，每个rank先做真实双卡保存/恢复短测，再正式训练；有完整断点时最多尝试两次自动恢复。


GPU0单卡切换：rank16和32已完成双卡训练。rank64在首步因GPU1显存不足退出，无训练断点；按用户要求在GPU0从原始backbone启动。单卡batch4、累积4保持global batch16，数据、seed、epoch、LR和新方案保持；单/双卡随机计算不保证逐位一致。rank64先验证单卡保存恢复；每100步保存完整断点。四项评测随后在GPU0执行，输出eval_results_paper_gpu0。


并行评测：按用户要求，scripts/parallel_completed_eval.py 在GPU0上串行评测已完成的rank16/32，同时rank64继续训练。只有一个评测进程运行；正式评测脚本使用全局文件锁及各数据集成功标记，避免与主队列重复执行。单独状态记录于automation/orbit-paper-seed42/parallel-eval-status.json；题目、推理参数和评分保持。共用GPU可能降低训练和评测速度。


并发修订：按用户要求rank16与rank32评测同时执行，rank64训练继续；GPU0最多两个评测进程，各模型/数据集使用独立锁避免重复评测。单独rank32状态为parallel-eval-r32-status.json，实时表同时显示两轮评测。共用GPU使每个任务变慢，不保证总吞吐提升；题目、评分和生成设置保持。


并发回退：按用户确认，暂停rank32评测的进程组与调度进程，保留内存中的模型和已生成预测；rank16四项评测完成后，独立后台watcher自动恢复rank32，避免重算已有题目。暂停期间显存仍保留，但不再持续占用GPU计算。rank64训练继续，后续评测恢复串行。


磁盘配额恢复：账号200GB配额耗尽导致第600步保存和评测写入失败，已清理旧中间断点，仅保留各rank最近两个完整断点及最终模型。rank64已验证checkpoint-500模型/优化器/调度器可读取，将从500步继续；后续save_total_limit=2，每100步保存不变。已完成成绩保留，未完成评测重新执行。
