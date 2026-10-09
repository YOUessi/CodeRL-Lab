# TRAIN-007D — SFT 与 DPO 相同验证偏好对上的模型概率对照

**冻结日期：2026-10-10。状态：代码、CPU 测试和真实 Qwen Tokenizer 数据预检通过；GPU 模型分数仍未生成。** 本实验基于 TRAIN-007A 和 TRAIN-007C 已存在的真实模型权重，**只评估，不训练、不调整模型**。

## 研究问题

DPO 在 UltraFeedback 2048 train / 256 heldout 上完成了真实251次更新，最终 DPOTrainer `eval_rewards/accuracies=54.365%`。但那是 `β(log π_DPO - log π_ref)` 定义下的参考相对奖励，而不是对照组 SFT 在同一绝对评分标准下的“训练前准确率”；不能因为数值超过50%就宣称相对SFT真正改进。

因此同一冻结数据的252个有效偏好对要用同一个评价规则，对比 **SFT 训练后、DPO 开始前** 与 **DPO 训练后** 两份 Adapter。

## 输入与禁止信息泄漏

- Qwen/Qwen3-1.7B-Base pin `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`，同一NF4 double quant + BF16 backbone。
- SFT LoRA adapter SHA: `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`，来自4096 UltraChat SFT。
- DPO LoRA adapter SHA: `3cb6ff9bd51e273b8c77bcf07f574a708b1ad2b089e53f9b61f4dc265c93ced4`，来自正式 UltraFeedback 2048 DPO。
- 固定 UltraFeedback source `daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2`；`dpo_validation.jsonl` SHA `9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450`；完整256对，TRl默认1024-token keep_start过滤4对，**原位保留252对**，不以正确率筛选。
- 只用本地标签 chosen/rejected，不读任何 LiveCodeBench private test 或MBPP hidden answer。
- GitHub Actions 固定数据预检以 `TRAIN-007D paired SFT DPO heldout preflight` 运行，使用真实 Qwen Tokenizer 验证 `256→252`，尚无CUDA输出。

## 主指标与辅指标

对于同一个 `prompt` 与两个固定回答 `chosen,rejected`，冻结 Qwen3 Tokenizer、DPO `keep_start` 的 max_length=1024，并对 prompt token 做 mask。推理只评价 completion tokens，避免长prompt混入：

[
m_{\rm avg}(x)=
\frac{\log P_{\theta}(y^{+}\mid x)}{|y^+|}-
\frac{\log P_{\theta}(y^{-}\mid x)}{|y^-|}
]

主指标为 `m_avg>0` 的比例，**同一252题** SFT vs DPO paired accuracy 差，task bootstrap 20,000次 / seed42；同时报告两个模型的绝对正确数及 wrong→right/right→wrong。

辅指标为**总** completion log probability difference `m_sum>0` 的正确率。它与DPO的sum logps方法更接近，但存在回答长度偏置；必须与mean-token主指标同时报告，不选择看起来效果更好的一个。

当数据难度、prompt长度不同或reward margin有改变时，差异不能等同于真实人类偏好质量变化。尤其这项分析仍是对已经用于决定训练方案的同一heldout集的**事后比较**，不是新的前瞻性基准，更不是PPO。

## 运行与保护

- `src/coderl_lab/analysis/train007d_preference_eval.py`：使用同一个 NF4 Base，在 `PeftModel` 上挂载 SFT 与 DPO 两份只读 Adapter，逐题逐完成对比；GPU仅用于forward，评测不做 optimizer steps。
- 对输入源文件、冻结manifest、两份权重及训练摘要逐个SHA核验，输出 `frozen_inputs.json`。每条完成保存数值进度 `task_logps.jsonl`，异常退出后只能在身份完全一致时恢复，不能事后按得分筛掉题目。
- 只有成功完成252对 x SFT+DPO两臂才写 `summary.json`。
- `scripts/run_train007d_matched_eval.sh` 在进程启动前取得单卡 `flock`、检查GPU独占以及bitsandbytes依赖；只在 TRAIN-008A full reward-model完成后串行运行，避免争显卡。
- `scripts/queue_train007d_after_rm.sh` 是**Tang本机的可选一次性条件命令**，只能在Reward Model 2048/256完整模型与原始Adapter SHA全部核验成功后调用评测；缺少文件/其它GPU任务会拒绝。它不会向用户发提醒，也不是ChatGPT后台通知服务。
- 结果无论正负均保存，并且明确它不是 DPO reward metrics。

## 目前不允许声称的内容

- “DPO必然改善所有下游问答/编程结果”；
- “DPO训练loss或相对奖励准确率高于50%，因此比SFT更好”；
- “把验证偏好标签用于训练奖励模型之后，还能把它当作一个从未见过的正式新测试集”；
- “CPU模拟的配对改善就是GPU已经计算出的模型指标”。

本实验应完整结束、检查两份 adapter SHA 与所有252输入后，再得出实测结论。
