# 2026-10-17｜EXP-023 执行代理奖励臂训练验收

冻结7B SFT起点、训练题集、seed42、32 optimizer steps、4 generations/group，与格式奖励及完整奖励臂一致。

运行记录：Tang `/home/you/projects/CodeRL-Lab-track-train017/artifacts/exp023/grpo7b_execution_seed42/summary.json`。

- 32/32 真实 GPU GRPO 优化流程完成。
- 奖励调用128次，128次奖励均为1.0，奖励存在变化的任务组0。
- 与SFT adapter逐元素比较的变化数=0；该臂未产生实际策略更新。
- 训练时间288.71秒；训练后 adapter SHA256 `08d5ec82e1a1e421192a59ddb3f58cc9272315b5c78e00f1f62cad0c2f132f6f`，与格式臂相同。
- 执行代理的奖励定义是：静态代码合法且Docker没有顶层异常（verdict为none或wrong_answer）则得1.0。原判题驱动会把部分逐样本函数异常捕获并归类wrong_answer，这个奖励不能代表所有测试输入上真正无异常执行；不能将其称为程序正确奖励。
- 前40次奖励中30次实际上为wrong_answer，仍得1分，进一步解释其信号饱和。该规则对当前训练分布无效，但不等于所有可能的“执行奖励”都无效。

科研结论：仅能断言**本次特定格式奖励和执行代理设计均未产生有效更新**。不能归纳为所有格式/执行奖励方法均不能工作。
