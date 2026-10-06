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

# 正式结果

固定 39 道同题、每题 16 个已生成候选，共 624 candidates / policy。

## EvalPlus base tests

| Policy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| Base | 41.99% | 74.65% | 80.61% | **82.05%** | **32/39** |
| SFT | 48.24% | 65.98% | 69.06% | 69.23% | 27/39 |
| GRPO | **50.48%** | 68.27% | 72.30% | 74.36% | 29/39 |

## MBPP+ base + extra tests

| Policy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| Base | 34.78% | **61.83%** | **67.56%** | **69.23%** | **27/39** |
| SFT | 39.74% | 54.89% | 59.91% | 61.54% | 24/39 |
| GRPO | **40.54%** | 54.35% | 58.51% | 61.54% | 24/39 |

因此更强测试下仍然复现：

- SFT/GRPO 提高单次成功概率；
- Base 在多样本覆盖上更强；
- GRPO 相对 SFT 主要是小幅概率重分配，没有扩大 solved@16。

## 原始测试高估程度

候选级 robust retention = MBPP+ correct / EvalPlus base-test correct：

- Base：217 / 262 = 82.82%；
- SFT：248 / 301 = 82.39%；
- GRPO：253 / 315 = 80.32%。

即约 17%–20% 的“通过少量 base tests”候选，会被 MBPP+ 增强测试进一步淘汰。

平均每题成功率的 base→plus 落差：

- Base：7.21 个百分点；
- SFT：8.49 个百分点；
- GRPO：9.94 个百分点。

## 配对 bootstrap（39题，20,000次）

### SFT - Base：MBPP+ 经验成功率

- delta：+4.97 个百分点；
- 95% CI：[-3.04, +13.30]；
- P(delta>0)=0.8777。

### SFT - Base：plus solved@16

- delta：-7.69 个百分点；
- 95% CI：[-17.95, +2.56]；
- P(delta>0)=0.0472。

### GRPO - SFT：MBPP+ 经验成功率

- delta：+0.80 个百分点；
- 95% CI：[-0.96, +3.04]；
- P(delta>0)=0.7514。

### GRPO - SFT：plus solved@16

- delta：0；
- 39 题上 solved 集合数量没有变化。

## EXP-006C 结论

MBPP+ 的平均额外测试数约 105/题，明显强于原始 MBPP。

在固定模型输出不变、只提高测试强度后：

1. 所有模型的绝对正确率都下降，证明原始 MBPP 少量测试会高估程序正确性；
2. SFT/GRPO 的 Pass@1 仍高于 Base；
3. Base 的 Pass@16 / solved@16 仍高于 SFT/GRPO；
4. GRPO 相对 SFT 仍主要表现为轻微概率重分配，而不是覆盖扩张。

因此 EXP-006A/006B 的主要现象不是由原始 MBPP 测试过弱造成的。

下一步可以进入 EXP-004：过程级可验证奖励 / 代码完整性信用分配，同时保留 LiveCodeBench 作为后续时间更新外部分布验证。
