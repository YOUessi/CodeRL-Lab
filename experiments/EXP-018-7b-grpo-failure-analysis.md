# EXP-018：TRAIN-017 7B GRPO 训练后失误/覆盖机制审计

**2026-10-15，事后只读分析（retrospective），不更改模型/数据，不重新推理或判分。**

动机：TRAIN-017 已在新的24个参数级留出任务上、两个预冻结评测种子各4采样、三个7B状态（SFT、TRAIN-016 GRPO16、TRAIN-017 GRPO32）冻结576条生成与Docker判分。新GRPO相对SFT正确候选85/192 vs 75/192（+5.21pp，按题24组bootstrap95%CI[+1.04,+9.90]pp），但四采样至少一解仅32/48 vs31/48（CI包含0）。必须追问这10个额外正确代码是否集中于原本已能解决的题，是否受seed、生成长度、Docker超时、语法/入口异常等影响；不能直接声称解决新题更多。

数据源冻结：`artifacts/train017/fresh24_3arm_2seed_n4/generation_checkpoint.jsonl`（SHA256 `8960a134d2cdfcd850f93ce7106624906e337f8a53feff01f35d36dbb79efd92`）与 `docker_scored_checkpoint.jsonl`（SHA256 `57fe0efc279f856d46273defe0acc24bdd899de91441a5ba78720244fa356f7d`）；原 `generation_summary.json` SHA `8e21f47eef7662e328c968e0ebb81e599950c8ba705ccdf537d8884a4eeeabab`，原 `docker_score_summary.json` SHA `3c0e0d22bbf093fbdf01e82f182f2149aa13bddc48aba64ea0621b2483234d23`。严格按24个任务聚类，不把同题双seed与每题4次采样当成192道独立题。原始代码/隐含测试仅 Tang 保留，Github只存聚合统计与SHA。

预定只读统计：
- 每种任务家族、每个seed、每个候选draw的位置的对错数与部分奖励；
- 对每个task×seed，SFT solved@4 vs 新GRPO solved@4 的得失翻转；在有解的题上，四采样正确候选数是否变多；
- 逐题（两个seed平均）正确候选比例的配对差和分布（受益/受损/不变）。
- 验证新模型是否因生成格式、语法/入口、256-token上限、Docker超时造成假下降或误判；基础设施错误不能自动记为程序错误。
- 识别候选程序 SHA 完全相同的占比和代码格式，但此类**结构多样性不等于算法语义多样性**，不得拿它冒充真正探索覆盖。
- 对比独立SHA固定的机器结果，必须100%再现旧评分总数；若不一致拒绝输出。
- 不基于此24题调训练超参后还把该集合当新的独立验证。文档标明已看过历史结果（并非新盲测）。

并行后续：开展全新来源外部编程测试 MBPP+ 的预冻结子集与独立Docker评测，优先只读取官方公开题干进行生成，待完整冻结样本后才打开测试代码。该外部集同样是历史公开基准，不能保证预训练完全没有见过。
