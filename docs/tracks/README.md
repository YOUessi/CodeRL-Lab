# 双主线：后训练实操与机制科研

本索引是 2026-10-08 决策的唯一入口；既有 EXP-001～006 的实验号与记录保持不变。

| 主线 | 代码分支 | 当前优先目标 | GPU 状态 |
| --- | --- | --- | --- |
| A 后训练工程 | `feat/posttraining-track-a-data-qlora` | 真正扩展指令/偏好训练集，重跑 SFT 的 NF4 QLoRA / BF16 LoRA 同数据对照，随后完成 DPO、奖励建模和 PPO/RLHF | 仅在 B 的实际 GPU 进程结束后训练 |
| B 机制科研 | `exp/livecodebench-verifier-gated` | 冻结 EXP-004J/K 的规则，跑 EXP-004L LiveCodeBench v6 175 题跨分布复现 | 当前占用 Tang 4090 |

## 不得混淆的三种数据

- A 的 `UltraChat train_sft` / `UltraFeedback train_prefs` 是真实训练数据；各自 `test_sft` / `test_prefs` 只用于保留验证。
- B 的 `MBPP test` / `LiveCodeBench v6` 是科研评测与泛化检验数据，不加入 A 的任何训练数据。
- 旧实验的 MBPP train 374 道仍作为可验证代码奖励 GRPO 基线。不能用 LCB private tests 计算训练奖励或决定解码 Gate。

## 全局工作规范

1. GitHub 保存代码、配置、预注册、摘要结果、研究决策、日记和失败记录；权重和原始大规模生成结果留在 Tang，不放进 Git。
2. 所有正式数据来源固定完整 revision，记录 SHA256、选样规则、来源、划分和数据许可。
3. A/B 在不同分支开发；GPU 只能串行训练/生成。Python CPU 单元测试、GitHub CI 和文档可以并行。
4. A 的训练必须有 train/validation 真正隔离；B 的官方 private tests 只能在所有 policy 决策冻结后读取。
5. 不为了正结果修改 B 已注册的 threshold、bias 或 window；A 新实验另立编号与比较基线。
6. 当前 A 的 RLHF / PPO / 奖励模型不应被写成“已实现”。先完成可重复的训练样本、训练和显存/质量对照，再推进下一算法。

## 主线 A 后续模块（与当前训练隔离）

- [TRAIN-008A Pairwise Reward Model / PR #29](https://github.com/YOUessi/CodeRL-Lab/pull/29)：在**基于主线A的独立叠加分支**中实现 UltraFeedback 2048/256、Qwen3-1.7B 的 Bradley–Terry 标量奖励模型、CPU审计及可恢复训练；当前尚未在 GPU 实训，因此不算已完成的奖励模型实验或 PPO/RLHF。
- 不在现有 PR #28 里混入 TRAIN-008A 源文件和大量未验证模型结果；只有父分支验收之后再独立审阅该后续 PR。

## 进度与入口

- A [TRAIN-007A/007B：H4 数据 + QLoRA/LoRA](../../experiments/TRAIN-007A-posttraining-data-qlora.md)
- B [EXP-004L：175题冻结外部分布](../../experiments/EXP-004K-heldout-verifier-gated-replication.md)，当前完整 L 分支文档位于 `exp/livecodebench-verifier-gated`。
- A 日志：`docs/tracks/posttraining-a/daily/2026-10-08.md`
- B 日志：`docs/daily/2026-10-08.md`（B 分支独立更新）
