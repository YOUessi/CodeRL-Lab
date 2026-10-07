# EXP-004A：执行阶段可验证奖励 / 代码完整性信用分配

## 状态

- 奖励设计：已实现第一版
- 静态依赖分析：已修复嵌套作用域误报
- 固定输出离线诊断：已完成
- 1.7B SFT 全训练集 4-sample 奖励方差审计：运行中
- 2-step GPU 冒烟：待离线审计通过后运行
- 187-step 正式训练：未开始

## 研究动机

EXP-006B 的大 k 分析表明，大多数所谓能力丢失可以被更大采样预算恢复。
唯一长期 0/64 的 mbpp_validation_0534 进一步发现：

- SFT 60/64 候选会使用 re.*；
- 但 60/60 缺少 import re；
- 只补 import re 后 17/64 直接通过隐藏测试。

因此当前具体问题不是模型完全不会算法，而是模型知道核心操作但输出代码不完整，导致运行阶段失败。

## 对照原则

基线：EXP-006A 的 1.7B 纯 GRPO。

唯一主变量：

旧奖励：语法 + 公共测试 + 全通过奖励。
执行阶段奖励：语法 + 依赖完整性 + 运行时干净 + 公共测试 + 全通过奖励。

其它保持不变：

- Qwen3-1.7B-Base 固定 revision；
- 同一个 EXP-006A SFT adapter；
- 同 train split；
- num_generations=4；
- 187 optimizer steps；
- beta=0；
- loss_type=grpo；
- 相同学习率、batch、seed；
- hidden tests 永不进入训练 reward。

## 第一版执行阶段奖励

权重：syntax 0.05，dependency_complete 0.10，runtime_clean 0.10，public_test 0.65，all_public_pass 0.10。

依赖完整性使用 Python symbol table 做作用域分析，避免把 lambda 参数、comprehension 变量、nested helper 参数和递归 helper 误判为缺依赖，同时仍能检测 re、math、Counter 等真实缺失依赖。

运行时干净信号将 AssertionError 视为正常逻辑判定，将 NameError、TypeError、IndexError、timeout、crash 视为运行阶段失败。

## 固定输出离线诊断

在 EXP-006B n=16 的 4320 个 validation completions 上，修复作用域误报后：

Base：hidden-correct dependency complete 100%，hidden-incorrect 56.13%；hidden-correct runtime clean 100%，hidden-incorrect 45.98%。
SFT：hidden-correct dependency complete 100%，hidden-incorrect 74.64%；hidden-correct runtime clean 99.81%，hidden-incorrect 60.24%。
GRPO：hidden-correct dependency complete 100%，hidden-incorrect 77.07%；hidden-correct runtime clean 99.82%，hidden-incorrect 63.11%。

高频 unresolved names 包括 re、math、gcd、heapq、cmath、groupby、Counter。
高频 runtime failures 包括 NameError、TypeError、IndexError、SyntaxError。

因此依赖完整性和运行时干净都与 hidden correctness 存在明显区分力。

## 训练前强制门槛：奖励方差审计

固定 1.7B SFT policy，对全部 374 个 train task 各采样 4 次，共 1496 completions。
对同一批 completions 同时计算旧 outcome reward 和新 execution-stage reward。

重点比较：

- outcome flat-group fraction；
- process flat-group fraction；
- 多少 outcome-flat prompt 被过程奖励变成 mixed；
- 是否有 outcome-mixed 被错误压成 flat；
- rescued flat 主要来自 dependency 还是 runtime 信号。

只有过程奖励能明显减少 zero-std prompt，才进入正式 GRPO。

## 正式训练门槛

离线审计通过后：16 task / 2 step smoke → reward component 日志检查 → hidden-test 泄漏检查 → OOM / Docker 泄漏检查 → 187-step 正式训练。

## 正式评测

统一使用 EXP-006B 口径：MBPP validation 90 题、n=16、Pass@1/4/8/16、hidden mean、solved@16、syntax/runtime/dependency diagnostics、zero-grad / frac_reward_zero_std、配对 bootstrap。

最终回答：更细的执行阶段可验证奖励，能否提高真正隐藏测试正确性与代码完整性，而不仅仅让训练 reward 更密？

## 离线审计预注册通过门槛

为避免看到结果后再调整判断标准，在全训练集 1496 个固定 rollout 完成前预先规定：

- process flat-group fraction 相比 outcome flat-group fraction 至少下降 10 个百分点；
- 或者 process reward 至少救活 20% 的 outcome-flat prompt；
- mixed→flat 的组不能超过全部 prompt 的 5%；
- 审计只使用 train split 公共测试和静态/运行时训练可见信号，不使用 hidden test。

满足前两项任一项、且不违反 mixed-collapse 限制，才进入 2-step GPU smoke 和 187-step 正式训练。

## 全训练集 1.7B SFT 奖励方差审计结果

固定策略：Qwen3-1.7B-Base + EXP-006A SFT adapter。
374 个 train task × 4 samples = 1496 completions。

生成参数：temperature=0.8，top-p=0.95，max_new_tokens=256，seed=42。
固定 predictions SHA-256：e679eddfdfb371bbbe0d0c924011069e55a1aceb75884f7633135effa0d2a67b。

生成耗时 580.16 秒；峰值 reserved 3,837,788,160 bytes。

### Flat-group 对照

- outcome flat：208 / 374 = 55.61%
- process flat：161 / 374 = 43.05%
- 下降：12.57 个百分点
- outcome-flat 被救活：47 / 208 = 22.60%
- mixed→flat：0

因此同时满足预注册门槛：

- flat fraction 至少下降 10 个百分点：满足
- 或救活至少 20% 旧 flat prompt：满足
- mixed-collapse 不超过 5%：满足，实际为 0

### 被救活组的信号来源

47 个被救活的 prompt 全部原 public-test reward 为 0。

- dependency variation：28
- runtime variation：47
- syntax variation：0
- runtime-only：19
- dependency + runtime：28
- dependency-only：0

这说明第一版 shaping 真正增加组内差异的直接来源是运行阶段差异；依赖完整性主要与 runtime failure 共变。后续若 EXP-004A 有收益，需要谨慎解释，不能把收益单独归因于 dependency 静态项。

### 总体信号

- dependency complete rate：81.15%
- runtime clean mean：74.13%
- public pass mean：45.69%
- outcome reward mean：0.5081
- process reward mean：0.5449

结论：离线奖励方差审计通过，可以进入 2-step GPU smoke。

## 2-step GPU smoke

16 个 train task，2 optimizer steps，num_generations=4，纯 GRPO，beta=0。

结果：

- reward mean：0.3828
- reward std mean：0.3881
- frac_reward_zero_std：0.25
- zero-grad steps：0 / 2
- dependency_complete mean：0.6875
- runtime_clean mean：0.5625
- public reward mean：0.28125
- peak allocated：4,931,461,120 bytes
- peak reserved：6,362,759,168 bytes
- train runtime：10.27 s
- Docker 残留容器：0

Smoke adapter SHA-256：f4c2cffe61afa93653a3c102c8c1b430438fecebf40dc54db27eccbe35064a47。

结论：GPU smoke 通过，可以进入与 EXP-006A 纯 GRPO 完全同预算的 187-step 正式训练。

## 离线奖励方差审计结果

固定 1.7B SFT policy，对 374 个 train task 各采样 4 次，共 1496 completions。
同一批 completions 同时计算旧 outcome reward 和新 execution-stage reward。

结果：

- outcome flat tasks：208 / 374 = 55.61%；
- process flat tasks：161 / 374 = 43.05%；
- flat 比例下降：12.57 个百分点；
- outcome-flat 被过程奖励救活：47 / 208 = 22.60%；
- rescued zero-public tasks：47；
- outcome-mixed 被压成 process-flat：0；
- overall dependency complete：81.15%；
- overall runtime clean：74.13%。

47 个被救活的 flat group 中：

- dependency variation：28；
- runtime variation：47；
- runtime-only：19；
- dependency + runtime：28；
- syntax variation：0。

预注册门槛判定：

- flat 比例下降至少 10 个百分点：通过（12.57）；
- 或救活至少 20% outcome-flat：通过（22.60%）；
- mixed→flat 不超过 5%：通过（0%）。

因此 EXP-004A 允许进入 2-step GPU smoke。

## 187-step 正式训练结果

固定与 EXP-006A 1.7B 纯 GRPO 完全相同的训练预算：374 tasks、187 optimizer steps、4 generations、1496 RL completions、beta=0、loss_type=grpo。

唯一主变量是 reward：outcome-only → execution-stage v1。

### 训练动力学

| 指标 | 纯 GRPO | 过程奖励 GRPO | 变化 |
|---|---:|---:|---:|
| zero-grad steps | 60 / 187 | **34 / 187** | -26 |
| zero-grad fraction | 32.09% | **18.18%** | **-13.90 pp** |
| mean frac_reward_zero_std | 60.16% | **43.05%** | **-17.11 pp** |
| effective steps | 127 | **153** | +26 |
| completions / effective step | 11.78 | **9.78** | -2.00 |
| public reward mean | 44.69% | **45.62%** | +0.94 pp |
| entropy mean | 0.2452 | 0.2424 | -0.0029 |
| train runtime | 815.22 s | 841.91 s | +26.69 s |

过程奖励新增组件均值：

- dependency_complete：83.36%；
- runtime_clean：76.91%。

前25%→后25%：

- public reward：39.40% → 54.48%；
- dependency complete：86.41% → 89.40%；
- runtime clean：77.99% → 85.19%；
- process zero-std：43.48% → 47.83%。

因此过程奖励显著减少整体无效更新，但 zero-std 并没有在训练后段继续单调下降；它主要通过把原本 outcome-flat 的组拆分成有执行阶段差异的组，提升整个训练周期的有效梯度密度。

正式 adapter SHA-256：571052fe0e43eaf9a2b1900d3b11dadc2a137ae60e38f8d2ea95c7a96eb5eec2。

峰值 GPU：allocated 5.15GB，reserved 9.20GB；Docker 残留容器 0。

训练效率层面已经通过。当前 n=16 validation 正在运行，只有 hidden-test / Pass@k / execution-stage diagnostics 同时改善，才能宣称 EXP-004A 方法有效。

## 正式 n=16 validation：matched pure GRPO 对照

公平基线使用 EXP-006B 的同一个 1.7B pure GRPO n=16 评测，而不是 EXP-006A 的 4-sample 旧评测。

| 指标 | 纯 GRPO | 过程奖励 GRPO | 变化 |
|---|---:|---:|---:|
| Pass@1 | 38.82% | **39.72%** | +0.90 pp |
| Pass@4 | 56.85% | 56.87% | +0.02 pp |
| Pass@8 | **63.63%** | 62.37% | -1.26 pp |
| Pass@16 | **68.89%** | 66.67% | -2.22 pp |
| hidden mean | 41.32% | **41.56%** | +0.24 pp |
| solved@16 | **62** | 60 | -2 |
| 16/16 全正确任务 | 9 | **11** | +2 |
| 语法失败 | 8 / 1440 | **4 / 1440** | -4 |

### 代码完整性 / 运行时诊断

| 指标 | 纯 GRPO | 过程奖励 GRPO | 变化 |
|---|---:|---:|---:|
| dependency incomplete | 202 | **185** | -17 |
| runtime unclean | 326 | **314** | -12 |
| NameError | 207 | **198** | -9 |
| TypeError | 80 | **75** | -5 |

两条 policy 的 hidden-correct 候选 dependency complete 都是 100%；过程奖励主要减少 hidden-incorrect 区域中的依赖/运行时失败。

### Pass@k 配对 bootstrap（20,000次）

过程奖励 - 纯 GRPO：

| 指标 | delta | 95% CI | P(delta>0) |
|---|---:|---:|---:|
| Pass@1 | +0.90 pp | [-0.42, +2.29] pp | 0.8947 |
| Pass@4 | +0.02 pp | [-1.74, +1.79] pp | 0.5100 |
| Pass@8 | -1.26 pp | [-4.05, +1.43] pp | 0.1811 |
| Pass@16 | -2.22 pp | [-7.78, +3.33] pp | 0.1485 |

所有区间都跨 0，因此没有证据认为 execution-stage reward 提高了总体 Pass@k。

### 训练效率结论

过程奖励对训练动力学的改善是明确的：

- zero-grad：32.09% → 18.18%，下降 13.90 个百分点；
- mean frac_reward_zero_std：60.16% → 43.05%，下降 17.11 个百分点；
- effective optimizer steps：127 → 153，增加 26；
- 每个有效 step 的 RL completions：11.78 → 9.78；
- dependency incomplete validation candidates：202 → 185；
- runtime unclean：326 → 314；
- 代价是训练时间 815.22s → 841.91s，增加约 26.7s。

因此第一版 execution-stage reward 确实让 RL 更新更密、更少浪费，并轻微改善代码完整性。

### 能力结论

但 matched n=16 validation 显示：

- Pass@1 只有小幅正向点估计；
- Pass@4 基本不变；
- Pass@8 / Pass@16 反而轻微下降；
- solved@16 从 62 降到 60；
- paired bootstrap 没有任何 Pass@k 显著改善。

所以当前最准确的结论是：

> execution-stage reward v1 成功解决了部分 credit sparsity / flat-group 问题，也减少了一部分依赖与运行时错误，但这些训练效率和代码完整性改善尚未转化成统计可靠的总体能力增益。

这是一条重要负结果：更密的 reward signal 不等于更强的最终 policy。

## 下一步

不继续调权重后在同一 MBPP validation 上刷结果。

下一步先做外部强测试重判：

1. 固定当前 n=16 process-GRPO completions；
2. 使用官方 EvalPlus / MBPP+ 测试；
3. 与 EXP-006B pure GRPO 的同一 39 个 MBPP+ 任务做配对比较；
4. 如果 process reward 在 MBPP+ 仍无改善，则 EXP-004A v1 结束，转向更有针对性的 runtime-repair / dependency-aware data intervention，而不是继续堆 reward 权重。

## MBPP+ / EvalPlus 外部强测试

不重新生成输出，直接使用当前 process-GRPO 的固定 n=16 validation completions。
与 EXP-006C pure GRPO 使用相同 39 个可映射 MBPP+ 任务、每题16候选、EvalPlus官方测试。

| 指标 | 纯 GRPO | 过程奖励 GRPO | 变化 |
|---|---:|---:|---:|
| Plus Pass@1 | 40.54% | 40.22% | -0.32 pp |
| Plus Pass@4 | 54.35% | 54.37% | +0.02 pp |
| Plus Pass@8 | 58.51% | 59.06% | +0.55 pp |
| Plus Pass@16 | 61.54% | 61.54% | 0 |
| Plus solved@16 | 24 / 39 | 24 / 39 | 0 |
| mean robustness gap | 9.94% | 9.46% | -0.48 pp |

配对 bootstrap：

- Plus empirical success delta：-0.32 pp；95% CI [-1.76, +0.96] pp；
- Plus solved@16 delta：0；
- robustness gap delta：-0.48 pp；95% CI [-1.92, +0.80] pp。

外部强测试仍没有 process reward 带来能力增益的证据。

## EXP-004A v1 最终结论

第一版 execution-stage reward 得到了三个明确结果：

1. 训练信号利用率改善成立：
   - zero-grad 32.09% → 18.18%；
   - mean frac_reward_zero_std 60.16% → 43.05%；
   - effective steps 127 → 153。

2. 代码完整性改善成立但幅度有限：
   - dependency incomplete 202 → 185；
   - runtime unclean 326 → 314；
   - syntax failures 8 → 4。

3. 总体能力增益不成立：
   - MBPP n=16 的 Pass@1/4/8/16 bootstrap 全部跨0；
   - solved@16 62 → 60；
   - MBPP+ Plus Pass@k 基本不变；
   - 外部 Plus empirical success 的 bootstrap 区间跨0。

因此：

> 更密、更细的执行阶段 reward 确实减少了训练浪费，但“更密的 credit signal”并不会自动转化成更强的最终 policy。

这意味着继续调 reward 权重的研究价值已经很低。

## 下一步：EXP-004B 局部运行时修复偏好学习

下一阶段不再调整 reward shaping，而直接针对已观察到的真实 failure mode：

- missing import；
- NameError；
- TypeError / helper omission；
- 逻辑大体正确但程序不完整。

核心思想：从模型自己的失败候选中构造“原始失败代码 vs 最小可验证修复代码”的近邻偏好对，只学习局部代码完整性修复，而尽量不改变算法主体和探索分布。

优先考虑 DPO / pairwise preference，而不是再次大规模 SFT，以减少 EXP-006A 已观察到的 SFT 覆盖收缩风险。
