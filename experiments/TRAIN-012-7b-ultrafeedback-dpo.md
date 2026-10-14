# TRAIN-012：Qwen2.5-7B 的 UltraFeedback DPO 偏好优化

## 2026-10-14 冻结配置，先验证再执行

**A 线工程目标：** 从已完成的 TRAIN-011 7B UltraChat QLoRA-SFT 继续，利用原先完整扫描且冻结的 UltraFeedback 2048偏好训练对 + 256对独立留出集，真正执行 DPO（直接偏好优化，Direct Preference Optimization）梯度更新。与 B 线的 EXP-7C/7D 独立，不把 HumanEval+ 评分用于挑选训练参数。

### 固定模型和输入
- Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`，NF4 double quant，BF16。
- 冻结 TRAIN-011 SFT Adapter SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`，SFT 4096 条正式训练，1 epoch，adapter save SHA与运行摘要必须一致。
- 偏好数据来自 `HuggingFaceH4/ultrafeedback_binarized` 修订版 `daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2`，`train_prefs` 2048对，`test_prefs` 留出256对，无 prompt 重叠。
- 训练集 SHA `945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77`；留出 SHA `9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450`。
- 重用经过 TRAIN-007C 真机跑过的本项目 `src/coderl_lab/train/dpo.py`：可训练 policy 是 SFT LoRA，隐式参考 reference 禁止更新；如果发现 TRL 版本差异须在 smoke 登记，不可宣称凭借 DPO 名称完成训练。

### 本次固定工程参数
- `configs/train012_7b_qlora_dpo_ultrafeedback.yaml`：BF16+NF4, LoRA r8 (继承 TRAIN-011), `beta=0.1`, `loss_type=sigmoid`, learning rate 1e-6, `gradient_accumulation_steps=8`, `per_device_train_batch_size=1`, `max_length=512`, `seed=42`。
- 正式目标 1 epoch / 2048偏好数据/256留出，因单卡 16GB 的计算图尚未验证，**现在不提前启动正式2048全量任务**。
- 第一道 GPU 门槛：32对固定训练偏好 + 8对留出、2 optimizer steps，必须同时通过来源SHA、严格分离、GPU独占锁和不少于12GB空闲显存。
- 验收：在 Tang 真机实际完成正向、偏好loss、反向、梯度更新、冻结参考与 Adapter 保存，报告实际loss、梯度、eval/rewards指标及 GPU 峰值。如果报 OOM、版本/API不兼容或输出缺失，留下失败日志和原因，不以部分完成冒充成功。
- Smoke 通过后再决定2048对正式训练是否满足显存与耗时要求；正式效果仍需**从未用于配置选择的独立比较**，训练损失不能替代人类偏好改善。
- 仅向 GitHub 提交配置、程序、每日事故日志、可复现统计与模型/数据SHA；训练集文本及原始模型权重仅存 Tang。

### DPO 与现有两条线的关系
- A 线：真实 SFT → DPO 工程闭环的更大模型对照。
- B 线：在独立试验中观察偏好对齐是否重塑生成正确率/结构多样性，避免只看 UltraChat NLL。已看过的 HumanEval+ 结果只能用作事后诊断，不可倒过来调本次 DPO。
