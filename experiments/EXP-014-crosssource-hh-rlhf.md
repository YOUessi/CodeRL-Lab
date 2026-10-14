# EXP-014：新的跨来源人类偏好验证与三模型盲审生成

**冻结日期：2026-10-15；本文件在任何本批数据的模型生成或偏好评分前提交。**

## 研究目的
延续 TRAIN-012/013 的结论：7B DPO 237 步下，FP32 LoRA 比 BF16 更新更充分，在此前 UltraFeedback 留出对上提升了隐式奖励边际，但偏好二元判断正确率未变。需测试这一差异能否迁移到**此前 CodeRL-Lab 未使用的另一来源人类偏好**，并产生可供人工盲审的真实模型生成回答。不在已经反复分析的旧256对 UltraFeedback 上调参。

## 冻结数据与风险
- 来源：`Anthropic/hh-rlhf` 的 `helpful-base/test.jsonl.gz`，固定revision `65649da1780c54952392cb81400f0e9684641333`；原文件 SHA256 `8be3fc1a13b27901631696f2be6f184c799f1baeaf145f53ac5db24960adc37b`（Tang 实际下载并验收）。
- 原记录2354对，仅保留chosen/rejected对应**完全相同的最后一个Assistant轮次之前的上下文**、Human/Assistant轮次交替、回答非空且不相同的样本。未经验证的错位对一律丢弃并计数，不能针对模型正确性挑选。
- 为减少内容截断偏差，仅在提示长度不超过448 token（固定Qwen2.5-7B-Instruct tokenizer+原生聊天模板）的合法对里，用 `SHA256('EXP-014|seed=42|'+canonical_context)` 稳定排序，取前 **256对**。去重和与旧 UltraFeedback 2048训练对/256旧验证对的精确归一化比对在取样前进行，不允许后续更改顺序。前64个固定提示用于三臂生成盲审包。
- 模型生成阶段只能读取 `public_prompts.jsonl`，绝不读取 `private_pairs.jsonl` 中的偏好标签、chosen/rejected 文本；两个文件都只在 Tang 本地保存，GitHub仅保存哈希、来源、计数与脚本。
- 来源数据是公开历史偏好集，不是 2026 新的时间前瞻集，不能排除预训练污染；跨来源也不能直接证明高质量或普适对齐改进。HH helpful-base 的标注分布与 UltraFeedback、基准提示形式不同，必须按域迁移理解。

## 三个冻结模型状态
1. 初始 SFT：Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` + TRAIN-011 SFT Adapter SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`。
2. BF16 DPO：相同7B骨干 + TRAIN-012 BF16 237步 Adapter SHA256 `d586d9ef32377189aa7d79b70ea5e37b5d5f26b930a4f60266ca024213f1a185`。
3. FP32 DPO：相同7B骨干 + TRAIN-013 FP32 237步 Adapter SHA256 `471703e1cec7fd322219133c10e09fa62aab2f6950b896688c7c073de1543281`。
- 三组共享相同Qwen tokenizer、HF chat template、NF4 double quant BF16骨干计算、同一提示文本、输出长度预算与逐题固定随机种子。推理中 PEFT adapter dtype需要记录并统一，避免以加载dtype误冒充训练精度影响。

## 独立偏好对数概率协议（256对）
- 将完整历史对话转换为 `system`无、Human→user / Assistant→assistant、最终以 assistant 生成提示结束。原 chosen/rejected 最后一轮回答作为单独候选，只做 teacher-forced 条件 logP，不是模型生成。
- 固定最大总序列长度 **768 tokens**；提示预算 **448 tokens**；对两候选各自严格记录在长度窗口内的可评分回答token数量。任何一方没有可评分回答token的样本计为不可评分并计入覆盖率，不按正确性/分数筛选。
- 同题三模型执行相同序列teacher-forcing。计算每对 `logP(chosen)-logP(rejected)`：总和logP的原始偏好准确率及**按回答token归一化**的偏好准确率（两个并列指标，避免用不同答案长度产生虚假胜负）；还统计 `β=0.1` 的隐式奖励边际相对SFT及FP32-BF16差值、二元选择正确/错误翻转。
- 对全部256对做20,000次配对Bootstrap（seed42，按偏好对重采样），输出准确率差与奖励边际差95%置信区间和有效样本数量。**不按这批结果回头调learning rate、beta、prompt或训练步骤。**

## 真实生成与人工盲审材料（64个prompt×三模型）
- 将上面冻结的256条 `public_prompts` 中稳定排序的前64条，用三臂相同聊天模板各生成一次：temperature=0.7、top_p=0.9、max_new_tokens=224、固定task_id种子42映射；不读取chosen/rejected，也不看偏好标签决定抽样。
- 所有 192 条输出及SHA保存在 Tang；按照另一固定随机映射，为每题生成A/B/C匿名乱序候选和单独密钥映射文件。**没有人工评分时只报告“已生成待审包”，不凭模型自身生成相似度或自评伪造客观质量分**。
- 若存在时间或内存约束，可先跑固定的前4任务工程烟测（独立目录），验证推理/dtype/生成完整度，再执行正式192条。真实人类盲评有观察者偏差，最终需至少2个相互独立的人类审阅者才能稳定声称“生成质量提高”。
- 原始文本、参考答案、生成回答不公开写入 GitHub，以免误扩散其中可能存在的敏感内容；仅保留冻结来源和结果统计，不上传私密API凭据。

## 执行与研究边界
- 两线均保持独立：A线已完成TRAIN-012/013真实训练；B线在此仅开展新来源外部迁移评测，不重复训练模型。
- 所有代码、配置、测试、每日故障/修复记录、实际统计与摘要以 GitHub 为唯一事实源，Tang RTX4090 仅承担必须真机推理。单卡GPU任务严格串行，另建日期日志。
- 在真实数据和全部三模型冻结的评分/生成完成之前，不能声称模型在此来源上的偏好泛化改善或生成质量改善。
