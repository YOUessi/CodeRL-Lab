# EXP-015：DPO 训练/评测契约一致性审计（预先冻结）

日期：2026-10-15。状态：代码与模型数值复核前冻结。本项目此前 TRAIN-012/013 的偏好评测指标已被阅读，**本实验只做已知数据的事后实现审计，不得称作新独立盲测**。

研究问题：TRL 1.14.1 的 DPOTrainer 在非对话偏好数据中自动补 EOS，使用独立的 prompt/completion tokenization 与 completion_mask，并在每个样本 max_length=512、keep_start 下截断，损失的logP仅求和回答mask。此前自研诊断取 `prompt+response`、字符 offset 作为回答token判断，未附加 EOS，可能造成二元偏好差异和不可评分样本统计差异。首先用真实训练库代码复算**256条冻结 UltraFeedback 留出对**的 token/mask/截断差异，再使用同一量化模型与已冻结 SFT、TRAIN-012 BF16 DPO、TRAIN-013 FP32 DPO Adapter，核对完全相同的输入下 logP 差异，不能偷偷修改训练数据或模型。

固定版本：`Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`，`trl==1.14.1`、`transformers==5.18.0`、`torch==2.10.0+cu128`，max_length=512，truncation_mode=keep_start，非对话 prompt原文，回答chosen/rejected原文 + EOS。冻结UltraFeedback test prefs SHA256 `9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450`，SFT SHA `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`；BF16 DPO SHA `d586d9ef32377189aa7d79b70ea5e37b5d5f26b930a4f60266ca024213f1a185`；FP32 DPO SHA `471703e1cec7fd322219133c10e09fa62aab2f6950b896688c7c073de1543281`。

验收一（CPU）：固定256对逐条实际Tokenizer与TR L的 `_tokenize`、`DataCollatorForPreference`，比较旧新completion token集合、prompt+completion前缀不一致率、EOS加入后每条有多少有效回答Token、由于prompt>=512的不可评分覆盖率，不输出原始prompt与参考回答到GitHub。针对简易字符串编写单元测试，证明EOS、字符边界和keep_start处理差异。

验收二（GPU，仅验收一完成）：同骨干NF4 double quant BF16前向，三适配器单序列准确选取TRL completion_mask且右移一个Token；输出 `logP(chosen)-logP(rejected)` 及各组 implicit reward margin 的新旧差异、252条有效数等完整统计，按题Bootstrap。训练的偏好目标依赖**相对参考策略奖励差**，不能只看raw logP符号作为DPO真实目标。禁止训练、增补答案或选择性丢弃错误题。GPU任务按单卡独占调度。

科研边界：该实验是实现一致性与错误修复，不是新模型性能提高、不能把后验修复称作预注册成功；旧评测结果永远保存，新指标使用新目录与明确身份，实验日期、失败修复和结果按日期提交。