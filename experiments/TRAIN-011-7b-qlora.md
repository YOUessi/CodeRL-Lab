# TRAIN-011 — Qwen2.5-7B-Instruct 4-bit QLoRA

状态：独立 A 线分支已创建；训练尚未开始。

## 当前事实
- Tang 在 B 线 `exp/7b-capacity-baseline` 成功以 NF4 4-bit 加载 Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 单次简单函数生成测试：CUDA 运行成功，峰值模型分配显存 5696.1 MiB。
- 这不代表 QLoRA 训练已验收；梯度与优化器激活占用仍需预检。
- 不能将 Instruct 与旧 Qwen3-1.7B-Base 的效果差异直接归因为参数量。

## 独立训练门槛
1. 沿用 TRAIN-007A 冻结的训练/验证切分与输入规范，检查现有 UltraChat 数据哈希，防止重复样本/held-out 泄漏。
2. 固定基础模型 revision、4-bit NF4 + double quant、LoRA target modules、BF16/FP16 compute 和数据样本顺序。
3. 先做两步梯度和参数更新烟测，核查 loss 有限、非零梯度、checkpoint 保存/恢复与 16GB 显存占用。
4. 再执行正式 QLoRA 训练，固定训练预算，并和原始 7B Instruct 建立同任务基线。
5. 使用冻结盲测对照，报告生成正确率、格式失败率、回答长度、预算及运行错误；不能只看训练损失。
6. 含 GPU 的阶段只通过 Tang；所有代码、配置、实测摘要和失败记录写入 GitHub。

注意：在烟测真实完成前，不得报告 TRAIN-011 已训练成功。
