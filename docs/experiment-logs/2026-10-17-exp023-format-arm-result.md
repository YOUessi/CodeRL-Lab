# 2026-10-17｜EXP-023 格式奖励臂真实GPU训练验收

运行机器：Tang RTX 4090 Laptop GPU 16GB。训练器：`src/coderl_lab/train/exp023_7b_grpo.py`。冻结训练数据与SFT起点与TRAIN-017相同。
- 32/32步真实GPU优化流程完成，128次奖励调用。
- 所有奖励都是1.0（128/128），按任务聚合的奖励变异组数0。
- 与冻结原始SFT模型比较：`initial_SFT_parameter_elements_changed_after_grpo=0`。
- 训练耗时336.85秒。
- 输出适配器 SHA256 `08d5ec82e1a1e421192a59ddb3f58cc9272315b5c78e00f1f62cad0c2f132f6f`，与SFT文件摘要不一致不代表参数数值变化，精确逐元素对比为0。
- 这是“形式流程完成”，**不是产生有效强化学习更新**。不能把训练奖励100%当能力提升证据。
- 此臂无需单独推理就可以根据数值相同的适配器判断参数级没有变化；但不同推理序列、随机数状态、解码实现可能造成输出不完全相同，仍需评测时复查。
- 当前执行反馈臂已启动，完整奖励臂排队等待前者运行成功后启动。
- 执行反馈臂仅为无顶层异常代理奖励，存在逐测试内部异常被沙箱捕获的混淆，不可宣传为严格执行成功奖励。

本地原始路径：`/home/you/projects/CodeRL-Lab-track-train017/artifacts/exp023/grpo7b_format_seed42/summary.json`。
