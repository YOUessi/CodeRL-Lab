# 2026-10-19：EXP028 v3 历史训练奖励信号审计入口

## 目的与边界

本入口只读历史 `reward_attempts.jsonl` 和对应的冻结 `train.jsonl`，不运行候选代码、不调用 Docker、不加载模型、不重新评分。每个历史运行严格要求 32 组、每组 4 个候选，共 128 条按顺序连续编号的记录。家族名称从训练数据中的 `task_kind` 映射；输出不含源码、提示或测试正文。

已核对 `exp027_matched_reward.rewards` 与 `exp026_core.features`：完整 corrected reward 为 `round(0.1 * executable_fraction + 0.1 * dependency_fraction + 0.8 * semantic_fraction, 6)`。语法无效或加载失败的分量与总奖励均为零。依赖分量是观测到的“无依赖错误”比例，不代表依赖完整性证明。

## 四类组的互斥定义

- `all_correct`：四个**最终奖励**均为 1。
- `all_wrong`：四个**最终奖励**均为 0。
- `same_intermediate`：四个最终奖励相同，且位于 0 与 1 之间。
- `nonzero_variance`：四个最终奖励不全相同。

另行报告 `semantic_all_wrong`、`semantic_all_correct`、语义正确比例是否零方差，以及“该候选是否全部测试通过”的二元正确性是否零方差。尤其是四个候选语义全错但执行与依赖分量均为 1 时，总奖励均为 0.2，应属于 `same_intermediate`，不能计入最终奖励全零组。四个候选均未全通过，但正确比例不同，也可能提供非零最终奖励组内信号。

方差字段使用总体方差，另给出样本标准差供与 GRPO 的组归一化尺度对照；这只是奖励诊断，不是梯度重放，也不证明 heldout 改善。

## 完整性检查

拒绝：缺失必需字段、重复或乱序 attempt/group ID、缺组、候选数错误、组内混任务或混测试、未知任务、非 corrected 变体、无效哈希、NaN、奖励/分量不在 [0,1]、分量逻辑矛盾、总奖励与已审计公式不一致、截断 JSON、重复 JSON 对象键与空记录。

CLI 另核对既有非零方差组数：narrow_seed43 为 28/32、narrow_seed44 为 30/32、broad_seed43 与 broad_seed44 均为 19/32。此处是历史一致性断言，不是本次重新运行得到的新结果；若断言失败需调查，不能自动改写记录或阈值。

## 执行方式

须先由主代理提交代码，再在 Tang 执行，例如：

```bash
python -m coderl_lab.analysis.reward_signal_audit \
  --run-name narrow_seed43 \
  --reward-log artifacts/exp028_v3/narrow_seed43/reward_attempts.jsonl \
  --train-data artifacts/exp028/data/narrow/train.jsonl \
  --output-dir artifacts/exp030/reward_audit/narrow_seed43
```

`--train-data` 应填写 Tang 上实际冻结数据路径；上例仅说明接口。其余运行替换运行名、奖励日志和 narrow/broad 训练数据。输出目录必须不存在。

输出为 `audit.json`（总计、逐组、家族、输入与审计源码 SHA256）、`groups.csv`、`families.csv`。没有抽到的训练家族另列在 `unobserved_families`，避免把没有观察值误当零奖励。逐组结果保留四个候选的数值分量，方便复核归类，但不泄露候选或测试正文。

## 本地验证

标准库 unittest 共 6 项通过：四类互斥计数；最终奖励与语义/二元正确性的区分；损坏、缺失与重复记录拒绝；严格 JSONL；训练家族映射；不输出候选源码与测试正文。命令：

```bash
PYTHONPATH=reward-audit/src python reward-audit/tests/test_reward_signal_audit.py -v
```

截至本文件写入，只完成纯 CPU 合成数据验证；尚未读取或分析 Tang 的历史日志，不宣称真实历史四类分布已经计算完成。
