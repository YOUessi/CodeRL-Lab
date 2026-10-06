# EXP-006C：MBPP+ 更强隐藏测试复核

## 状态

接口核验：完成。
严格同题映射：完成。
2题 smoke：待运行。
39题 × 16 × 3 policy：待运行。

## 研究目的

EXP-006A / 006B 的主要结论来自 MBPP 原始少量测试：后训练主要重分配正确轨迹概率，并改变代码完整性，而没有可靠证据表明底层算法能力支持被删除。

本实验完全不重新生成模型输出，只把 EXP-006B 的固定 n=16 completions 交给 MBPP+ 更强测试重新判定。唯一改变的变量是测试强度。

## 固定输出

- 1.7B Base n=16；
- 1.7B SFT n=16；
- 1.7B GRPO n=16。

每个模型仍使用 EXP-006B 的同一批 90×16 completions。

## EvalPlus 固定版本

- Python package：evalplus 0.3.1
- MBPP+ dataset version：v0.2.0
- MBPP+ hash：ee43ecabebf20deef4bb776a405ac5b1

## 同题交集

CodeRL-Lab validation：90 题。
MBPP+ 当前清洗数据：378 题。

按原始 MBPP source task id 严格映射：

- overlap：39 题；
- entry_point 一致：39 / 39；
- 不在 MBPP+ 清洗子集：51 题。

39 题的 MBPP+：

- 每题 base inputs 平均 3；
- plus inputs 平均约 104.87；
- plus inputs 最少 18；
- 最多 143。

## 判定口径

采用 EvalPlus 官方口径：

- Base correct：base tests PASS；
- MBPP+ correct：base_status 与 plus_status 都 PASS。

计算固定 16 candidates 下：

- base Pass@1/4/8/16；
- plus Pass@1/4/8/16；
- base solved tasks；
- plus solved tasks；
- robustness gap。

## 解释边界

这是严格同题、更强测试复核，但只有 39 道交集题。

它能回答：原 MBPP 少量测试是否高估了 Base/SFT/GRPO 的代码正确性与支持覆盖？

它仍不能替代时间更新的污染敏感基准，后续还需要 LiveCodeBench。
