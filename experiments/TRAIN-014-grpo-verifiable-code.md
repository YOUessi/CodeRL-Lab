# TRAIN-014：1.5B 编程正确性奖励的 GRPO 强化学习实操

日期：2026-10-15，GPU训练前冻结工程协议。A线目标：**真正跑通模型生成 → Docker执行测试 → 计算可验证奖励 → GRPO梯度更新 → LoRA保存 → 冻结题目评测**，不以简单loss数字或模拟奖励充数。

选择 Tang 本地缓存的 `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，单卡 NVIDIA RTX4090 Laptop 16GB，Transformers5.18.0、TRL1.14.1，优先使用Qwen2.5-1.5B而不是贸然上7B多路采样。训练配置：BF16底座、PEFT LoRA(r8, alpha16, dropout0, q_proj/v_proj) FP32可训练权重（检查实际dtype），`use_vllm=False`；2个采样/题、温度0.8、top_p0.95、max_completion_length=96/128、beta=0、seed42、先2 optimizer steps GPU烟测，只有代码、reward、update、保存和GPU指标都通过才扩大步数。

数据：程序化生成严格确定性整数单参数 `solve(x)` 问题，train与holdout任务ID、模板参数/函数映射不重叠，源数据SHA冻结，train rewards只可访问train专用的多个数值测试。holdout永远不进入GRPOTrainer和reward函数。真实候选无论内容如何都在受限Docker `python:3.11-slim`（预记录本机镜像SHA、禁网络、只读、非root、无额外capability、CPU1、256MB、pids≤32、候选级超时）内运行；严禁在宿主机exec模型生成的源码。Docker来源和容器异常不能默默计为模型答案错误。

固定评分：第一步语法与solve入口检查；测试全过奖励1.0；仅语法与入口正常但测试未全过奖励0.1；不可解析或入口缺失奖励0.0。偏好的是**代码正确性**而非字符串与答案表面的相似度。不得按看到训练或保留题成绩改变奖励权重/题型/温度；任何尝试新条件单独做新实验版本。记录每组是否存在不同奖励（否则GRPO没有偏好梯度信号）、模型输出有效比例、统计奖励均值、梯度是否非零、Adapter哈希/差异、训练时长及峰值显存。

工程步骤：先创建/冻结任务、奖励Docker容器、CPU测试和基准生成器；验证完全由真实候选驱动并拒绝恶意shell访问；进行两步真实4090训练和可恢复日志；必要时排查OOM或TRL API版本差异；能跑通再扩展训练和评测。对holdout报告真实pass@1/pass@2及方差，不把这批简单合成题称作新的高难软件工程基准。与已有B线DPO SFT/覆盖分析独立分支，GPU优先测试1.5B。所有源码/配置/审计/失败和摘要写GitHub，权重与原始输出保存在Tang。