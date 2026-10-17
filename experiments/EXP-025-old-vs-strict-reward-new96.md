# EXP-025｜旧完整奖励 vs 新严格奖励的同题对照（2026-10-17）

## 目的
验证 EXP-024 的新严格奖励是否确实优于 EXP-021 的旧完整奖励，而不仅是与冻结 SFT 基线的差异。

## 固定条件
- 设备：仅 Tang GPU；模型 Qwen2.5-7B-Instruct 固定 revision。
- 评测：EXP-024 已冻结的第三批 MBPP+96 道题，原始数据 SHA256 `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`，公开数据 SHA256 `6438a8abdfc5ec19edd98f717e03d17b2c8a13f5fbe25f07a63db7ee030755f7`。
- 对照：同一 SFT 起点、同样的公开提示、4次采样和相同种子索引；旧完整奖励 seed43 adapter `02ff43a2aa674c58d25ae4691e7f2269689851fb17037282e5d89a82871d4aee`，seed44 adapter `15f778181c7e3f02e930057f33c1fc7a068bc15f9988d2d1671599bb01bb5365`。
- 新严格奖励 seed43 adapter `d17670683ae80cb14a8a9f60fd1cc81fc71c768fa2592a878e3d4ea81079b1b3`；seed44 adapter `f3a909a17c548bfeaac468d65d9a3c99d4bc44b39a1b629b2a41a90adca92564`。
- 候选程序在冻结之后才允许读取 private test，且只在固定 Docker 沙箱运行。

## 预注册比较
按题目做成对对照：SFT、旧完整奖励和新严格奖励各自的正确候选/384、首样本正确/96、四次至少一次解出/96、NameError、Error→Correct与Correct→Error翻转；bootstrap 的重采样单位为题目。两个训练种子不是384个独立训练重复。重点检验 **新严格奖励 - 旧完整奖励**，不能单凭与 SFT 的差值判定新方法优胜。

## 当前运行状态（记录时）
- 旧完整奖励 seed43 96题的生成与判题顺序任务已在 Tang 实际启动，生成检查点出现 18/768；seed44 排队接续。
- 两个旧模型检查点状态及 SHA256 已校验；当前不能公布旧 vs 新的结论。
- 对照代码 `src/coderl_lab/analysis/exp025_old_seed{43,44}_{generate,docker_score}.py`、顺序脚本 `scripts/run_exp025_old_reward_new96.sh` 已提交 GitHub。原有 EXP-024 判题结果保留，不重复评测。
