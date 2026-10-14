# TRAIN-013 — 7B DPO 中可训练 LoRA 的 BF16/FP32 数值精度对照

**2026-10-14 冻结协议；首次训练前记录。**

## 研究问题
TRAIN-012 在单卡4090上已完成237步7B DPO，但冻结的256对UltraFeedback评测未发现准确率提升。与初始SFT相比，最终112个BF16 LoRA tensor仅3.97%元素发生可观察的数值变化，相对参数L2位移0.0003866。判断该现象是否主要来自**BF16 可训练 LoRA 权重的更新量化**，而不是把所有结果归因为学习率、偏好数据或算法无效。

## 关键源码发现
TRL 1.14.1 在量化模型 `DPOTrainer.__init__` 中会强制执行 `param.data = param.data.to(torch.bfloat16)`。因此PEFT加载时的浮点精度和yaml中model BF16不能证明**训练可优化LoRA存储**的实际dtype。TRAIN-013仅在 DPOTrainer 创建并复制初始 SFT 到 `ref` 冻结 adapter 之后、优化器创建之前，把 `default` 中 requires_grad 的 LoRA参数统一设置BF16/FP32；其他权重不动。逐Tensor核查冻结ref与初始trainable相等、reference不可训练。新控制仅在独立分支，TRAIN-012运行代码与已冻结checkpoint不可覆盖。

## 严格单变量匹配
- 两组均Qwen2.5-7B-Instruct base revision `a09a35458c702b33eeacc393d103063234e8bc28`；同一初始TRAIN-011 SFT Adapter SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`；NF4 double quant + BF16 前向计算；TRL1.14.1、PEFT0.21.2、torch2.10.0+cu128。
- 同一2048 UltraFeedback偏好训练集 SHA256 `945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77`，未用于训练的256对SHA256 `9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450`。训练/留出不重叠。
- 两个matched32实验使用完全相同seed42、32个optimizer步骤、batch1、grad accumulation8、DPO beta0.1、sigmoid loss、learning rate1e-6、warmup ratio0.03、cosine scheduler、梯度检查点、max512。为了彻底避免此前Trainer自带eval中的大张量OOM，训练两组都**不在Trainer内部做留出集eval**。
- **唯一实验变量**：训练可优化`default` LoRA参数 dtype为`torch.bfloat16`或`torch.float32`；`ref`保持冻结且相同（其dtype和初始化需在日志记录）。
- 两组分别生成独立的训练Adapter/日志，先两步真实GPU冒烟（比较训练参数dtype、loss/grad/保存是否通过），再各32步正式比较；任何一组失败不得把另一组成功冒充配对成功。
- 核验：保存后112个LoRA tensor的dtype/参数数、与同一初始SFT的element changed fraction与relativeL2、每步梯度 norm/rewards margin、显存/吞吐，并报告两组精度路径。
- **独立偏好诊断**：相同256对已在TRAIN-012中用于检查，**不是新的盲测**。对两个32步Adapter可执行同条件冻结512 Token逐条logP偏好评估，但结果只作机制探索，不能用来对这256对继续调参然后宣称前瞻显著改善。
- 先前研究观察可能有多种解释：学习率低、训练任务与偏好数据不匹配、BF16离散更新小、参考策略初始化或算法动态。只更改参数dtype时如果FP32出现更大delta、但验证无改善，不能宣称偏好优化已成功。
- 所有配置、脚本、测试、源码、失败/修复记录和可复现统计以GitHub为唯一事实源；GPU真机只用Tang，两个GPU任务必须串行。科研分支只提草稿PR，未审阅前不合并。
