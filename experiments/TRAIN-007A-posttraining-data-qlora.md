# TRAIN-007A / TRAIN-007B：后训练数据 + NF4 QLoRA / BF16 LoRA

## 研究与工程目标（2026-10-08 预注册）

本阶段回到 CodeRL-Lab 原先的目标：真正掌握数据工程、SFT、低秩/量化训练以及可复现的训练/验证闭环；不把 EXP-004L 的推理门控实验混入新 SFT 训练。

主要比较：**同一冻结数据、同一基础模型、同一更新预算与 LoRA 结构，四位量化 NF4 QLoRA 对比 BF16 LoRA 的训练稳定性、验证损失、显存占用和吞吐。**

量化可能降低显存，但是否牺牲质量必须真实跑过同数据对照才能下结论。

## 冻结模型与数据

基础模型：`Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`。

### SFT

- 来源：`HuggingFaceH4/ultrachat_200k`。
- revision：`b7fe606ecdbf71e8537946a8d9de5ccf0f6da48b`。
- 正式只从 `train_sft` 中用 seeded SHA256 规则选 4096 个 prompt，`test_sft` 只选 256 个 hold-out。
- 每条多轮对话第一组 `user→assistant` 对作为首阶段单轮指令监督样本，避免悄悄变成多轮训练；完整多轮 SFT 另立对照，不能偷换。
- `format: raw` 只输入用户指令与 `### Assistant:` 前缀，**不注入 MBPP 的“请完成 Python 编程任务”模板**。
- `train` 与 `validation` 按 prompt SHA256 去重，禁止交集。

### DPO 数据准备（训练第二阶段）

- 来源：`HuggingFaceH4/ultrafeedback_binarized`。
- revision：`daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2`。
- 正式从 `train_prefs` 选择 2048 对，`test_prefs` 只选 256 对。
- 要求 chosen/rejected 具有逐字相同的上下文 prefix、不同且非空的候选 assistant 答案。
- 不与 MBPP test 或 LiveCodeBench hidden 数据混合。
- 注意 DPO 模型必须来自本阶段 SFT checkpoint，等 SFT 完成后**先记录适配器 SHA，再冻结 DPO 正式配置**。当前仅数据准备，不宣称已经完成新 DPO 训练。

### 数据审计与可复现

- 流式读取真实 HF 源数据，完整 SHA256 得分选样，而不是只截前4096行。
- 保存：源 revision、格式版本、种子42、扫描数量、有效/过滤数量、SHA256、精确 train/validation 记录数量。
- 大规模原始训练样本存 `data/generated/posttrain-h4-v1/`，不放 Git；脱敏后的精炼 manifest 摘要在真实执行后提交 `results/`。
- `--smoke-scan-limit` 明确标记 `full_scan: false`，禁止被正式训练脚本接受。
- 训练数据来自公开合成对话和偏好比较，不等于经过领域质量人工审核；数据污染和领域迁移需另作评估。

## 公平对照

| 变量 | TRAIN-007A | TRAIN-007B |
| --- | --- | --- |
| 模型 | Qwen3-1.7B-Base | 同一个 revision |
| 数据 | UltraChat 4096 / holdout 256 | 同一 SHA256 数据 |
| 精度 | NF4 + 双重量化 + BF16计算 | BF16，不量化 |
| LoRA | r16, alpha32, dropout0.05 | 相同 |
| 模块 | attention + MLP 7个投影 | 相同 |
| batch / grad-acc | 1 / 16 | 相同 |
| epochs | 1 | 1 |
| 最大长度 | 1024 | 1024 |
| lr / scheduler | 1e-4 / cosine | 相同 |
| seed | 42 | 42 |

两个实验不并发占用 Tang 4090。定量比较训练耗时、train/eval loss、梯度/NaN、显存峰值、训练可行性。若优化器与量化底层不同，要在结论中明示，不宣称严格同计算图。

## 工程产物（截至 2026-10-08）

- `src/coderl_lab/datasets/posttrain_h4.py`：正式 SFT/DPO 数据转换、质量审计、确定性选样、prompt hash 去重与划分隔离。
- `src/coderl_lab/train/sft.py`：兼容原 MBPP 指令格式 + `format: raw`，可选 NF4 + BF16 LoRA 训练、验证集 loss、完整 log history。
- `configs/posttrain_a_{qlora,lora}_qwen3_1.7b.yaml`：预注册两组训练配置。
- `scripts/build_posttrain_a_data.sh`：实际下载与构造数据；smoke/full 分开。
- `scripts/run_posttrain_a_qlora.sh`：正式数据 SHA 校验、完整扫描验证、GPU 独占门禁后真实训练。
- `tests/test_posttrain_h4.py`：离线 CPU 测试，不以未下载训练集冒充正式结果。

## 运行指令（完成 GPU 互斥检查后）

```bash
pip install -e ".[dev,data,model,qlora]"
bash scripts/build_posttrain_a_data.sh sft smoke
bash scripts/build_posttrain_a_data.sh sft formal
bash scripts/build_posttrain_a_data.sh dpo formal
bash scripts/run_posttrain_a_qlora.sh
```

数据准备可与 B 的 GPU 生成在工程设计上并行；GPU 训练必须等 B 完成后再运行。

## 验收标准

- [x] 两条主线有独立 Git 分支。
- [x] 数据源 revision、选样/去重/训练保护规则已落代码。
- [x] 新旧 SFT 格式兼容性测试编写完毕。
- [x] NF4 QLoRA 与 BF16 LoRA 同数据配置完成。
- [x] CI 相关 CPU / 真实数据构建流程通过。
- [x] 真实 H4 源数据下载与 full-scan manifest 固定（GitHub Actions #37725387735）。
- [ ] 真实 CUDA 上的 QLoRA 2-step smoke。
- [ ] 两臂完整训练、验证 loss、显存和速度比较。
- [ ] 从新 SFT checkpoint 开展 2048 对 UltraFeedback DPO 正式训练。
- [ ] 继续扩展奖励模型、PPO/RLHF；这些不是本实验已完成的内容。

## 结果解释边界

当前**只有实验设计、实现与 CPU 验证**，还没有新模型训练的质量或算力数字。禁止把既有 MBPP 374题的 LoRA/SFT 成绩冒充新的 UltraChat 4096题 QLoRA 成绩。

## 2026-10-08：真实训练数据的正式快照完成

GitHub Actions：[完整来源数据构建成功](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37725387735)；可下载冻结训练样本的 [Artifact](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37725387735/artifacts/11527757268)（有保留期限）。

| Stage | Train（来源完整扫描量） | Validation（来源扫描量） | Train SHA256 | Validation SHA256 |
| --- | --- | --- | --- | --- |
| SFT | 4096 (207865) | 256 (23110) | `f868096a21eb37249d06d318fb56ab3b0e3e99c4db442e54c43d6f42b565888b` | `a7f6e07f8569157fe5c3ae0deb65875c3427f9db3d6a5120379e1d9b10dae170` |
| DPO | 2048 (61135) | 256 (2000) | `945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77` | `9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450` |

两条数据线均为 full_scan=true，prompt SHA256 train/validation overlap=0。机器可读归档：`results/train007a-data/summary.json`。这些是真实筛选的数据，不等于模型已重新训练，也不能凭数据生成就声称性能提高。

下一步单卡真实 QLoRA GPU smoke 使用训练32条 / 保留验证8条，训练步/显存确认后才启动4096条完整 SFT；LoRA 控制与 QLoRA 禁止同时训练。
