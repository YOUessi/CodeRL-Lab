# EXP-018：7B GRPO 正确候选增长但题目覆盖停滞的逐题机制审计

**日期：2026-10-15。性质：TRAIN-017 已看过最终结果后的事后只读机制分析；不生成新的程序、不重新训练、不更改 Docker 测试。**

## 需要解释的真实现象
TRAIN-017 的三个模型状态均为同一 Qwen2.5-7B-Instruct 基座：TRAIN-011 SFT、旧 TRAIN-016 GRPO16、新 TRAIN-017 GRPO32（同一 SFT 起点但训练任务和步数不同）。冻结新算法任务 24 道×2个随机种子×每题4次采样×3模型 = **576 条**。Docker 真实判分后，三个状态的正确程序 75、76、85 /192；但是四次内至少有一条正确程序只由31/48增到32/48。在没有独立新样本前，不得把候选提升直接称作通用编程泛化。

## 只读数据与冻结身份证明
- 训练数据 `21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb`；冻结公有heldout `3f8768827ba8cba10d3555a58a3de5e69ec197a0192f1adb40ea05ae13c0e7a8`；私有test oracle `2cea9653952f7357a5f0ea549a32a94d36e3a0b46a9332d5b7d46ee84978a027`。
- 576条已冻结生成 `generation_checkpoint.jsonl` SHA `8960a134d2cdfcd850f93ce7106624906e337f8a53feff01f35d36dbb79efd92`；576个 Docker 判分 `docker_scored_checkpoint.jsonl` SHA `57fe0efc279f856d46273defe0acc24bdd899de91441a5ba78720244fa356f7d`；生成总结SHA `8e21f47eef7662e328c968e0ebb81e599950c8ba705ccdf537d8884a4eeeabab`。
- 模型源 SFT SHA `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`，旧 GRPO16 SHA `3d84428201f8802cdcde33cb64dd2c46ee4f2538de850cb2240b017ff2a237c8`，新GRPO32 SHA `47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4`；Docker固定镜像 `sha256:d3846bce6e97d0ef5c32fc023300d5e39251ad791e6fdfaee41d2ba75246766e`。
- 代码、指标与审计数据在GitHub，原始576条模型代码与逐条沙箱日志仅Tang本地；研究分支与GPU权重均不可覆盖。

## 固定的只读分析（在实现和运行前记录）
1. 将24题、2seed、4次采样、3组模型按 **(task_id,seed,sample_id,arm)** 严格匹配；核对输出内容 SHA、Docker评分引用 SHA、一致性的全部96个题×seed×sample ID（24×2×4），拒绝缺失/重复/越界。
2. 分训练前后分析 **错→对/对→错/均错/均对** 的候选级翻转，汇总成功计数净变化，并按六类算法、两个seed和样本index分层；特别统计在同一个题×seed已至少有一条正确时，额外正确回答如何改变“可靠性”却不改变 `solved@4`。
3. 分任务查看 `solved@1`、`solved@2`、`solved@3`、`solved@4`曲线；哪道题×seed 由0覆盖变成有覆盖，哪道由有覆盖变为0，净变化如何对上最终31→32。
4. 源代码完全相同数量、每个题×seed的4条生成中不同的答案 SHA（描述性“代码文本多样性”，不是语义上不同的算法思路）；生成长度/是否触顶256Token；`candidate_timeout` / `invalid_code` / `wrong_answer` 失败类型汇总。
5. 20,000次按**24个任务聚类**的配对Bootstrap（seed42），将两个seed视为同一任务的重复测量，不把576个样本当作独立研究题；可补充精确符号/二项检验，但均为**后验探索性分析**，不能滥称多个显著指标。
6. 原始实验中的“pass@4”必须明确称为 `solved@4`：四个观测候选至少一条正确，不是使用多个抽样无偏估计定义的Pass@k。
7. 不许根据这批分数选择训练轮数、随机种子、奖励/抽帧策略并再次宣称新盲测。真正新数据应在本分析结果之后另外冻结不同算法家族，保留新鲜度与学习规模为7B。

## 预期交付
`src/coderl_lab/analysis/exp018_frozen_failure_audit.py`、严格CPU单元测试、`results/exp018/...` 聚合机器结果与 `docs/daily/2026-10-15-exp018.md`。不执行新模型推理或GPU训练。
