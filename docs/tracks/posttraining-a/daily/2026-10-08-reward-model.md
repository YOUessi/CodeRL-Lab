# 2026-10-08：TRAIN-008A 奖励模型研发记录

## 为什么开始这一支线

用户确认 CodeRL-Lab 要继续并行推进完整后训练实操平台（A）与推理机制科研（B）。A 已有真实 MBPP SFT / GRPO / DPO 实训，现在又有 UltraChat QLoRA 4096/256 正式模型，配对 UltraFeedback DPO 2048/256 已冻结；但完整 RLHF 尚缺训练得到的 **Reward Model** 和 PPO 策略更新。

这次不是先设计“智能RL方法”找效果，而是补齐 RLHF 的最基本可复现训练组件。Reward Model 的数据确实已由 GitHub Actions full scan 生成，不是模型在当前会话编造的数据。

## 实际完成的源代码

1. 基于 `feat/posttraining-track-a-data-qlora` 创建叠加开发分支 `feat/posttraining-reward-model`，Draft PR #29 以 A 分支为 base，避免把尚未验收的 A 大量记录直接推到 main。
2. `configs/posttrain_a_reward_model_qwen3_1.7b.yaml` 冻结 Qwen3-1.7B Base + NF4/double quant + BF16 + LoRA r16；新增 sequence classification `score` head；UltraFeedback 2048训练对/256 heldout；学习率1e-5、1epoch、gradient accumulate8、maxlen768、seed42。
3. `src/coderl_lab/train/reward_model.py`：读取完整冻结 SHA/manifest 并拒绝划分泄漏；对同一 prompt+chosen/rejected 输出成对 reward，使用负 log-sigmoid margin + 0.001奖励幅值惩罚；训练前/后均评估 heldout 选择准确率、margin 和成对 loss。
4. 单 GPU训练实现真实 PyTorch AdamW/余弦调度器、LoRA+score head 可训练参数、按32 optimizer steps 保存完整 adapter+optimizer/scheduler/RNG 检查点及训练身份 SHA。支持显式 checkpoint resume，禁止假续训。
5. `scripts/run_posttrain_a_reward_model.sh`：默认 smoke 使用32训练/16验证；formal为完整2048/256；每次必须通过数据哈希和GPU独占检查，并共享 A/B 实验锁。
6. `tests/test_posttraining_reward_model.py` 测试负 log-sigmoid 数值稳定、偏好对同prompt、划分泄漏和快照 SHA、checkpoint身份失败保护等；GitHub CPU CI 已通过。
7. `experiments/TRAIN-008A-pairwise-reward-model.md` 正式预注册研究问题、数据版本、模型/优化配置、成功判据和解释边界。

## 尚未完成（不得虚报）

- Tang 单卡当前正在执行 TRAIN-007B 4096样本 BF16正式训练，B 的EXP-004N单卡GPU任务也已排队。因此 TRAIN-008A **尚未执行真实GPU smoke或完整奖励模型训练**。
- 此时不存在可报告的 Reward Model heldout accuracy、模型增益或 RLHF/PPO 算法结果。
- 下一步优先在GPU空闲后做32/16真实训练 smoke，验证量化score head和梯度；再做2048/256正式奖励模型。PPO/RLHF另立实验，不能视为本RM PR已经完成。
