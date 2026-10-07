# EXP-004G：Trajectory Susceptibility / 低 Margin 决策边界

## 状态

- 分支：exp/trajectory-susceptibility
- 设计：已预注册
- 轨迹 margin profile：待实现
- 12-arm susceptibility label：待实现
- GPU profile：待运行
- 正式统计：待运行

## 研究动机

EXP-004F 已定位出如下机制：

tiny local parameter perturbation
→ prompt 起点 next-token 分布只有微小变化，top1 仍 100% 一致
→ 自回归过程中遇到 low-margin / tie 决策点
→ argmax 被极小 logit 变化翻转
→ prefix feedback 造成 deterministic trajectory 级联分叉
→ stochastic sampling 再增加额外漂移。

四个代表 arm 的首次 greedy 分叉中：

- 约 56% reference top1 margin = 0；
- 约 79% margin ≤ 0.10；
- 约 95% KL ≤ 1e-2。

下一问题不再是“这种机制是否存在”，而是：

> 在不知道 perturbation 结果之前，仅看 SFT 自己的 greedy margin profile，能否预测哪些 task 更容易被极小参数扰动改变轨迹？

如果可以，就从机制解释推进到可预测的 trajectory susceptibility 指标。

## 固定数据与模型

- Qwen3-1.7B-Base 固定 revision；
- reference policy：EXP-006A SFT adapter；
- MBPP validation 90 tasks；
- SFT greedy predictions：复用 EXP-004F 正式 greedy；
- susceptibility outcome：复用 EXP-004F 的 12 个 matched-perturbation greedy arms；
- 不训练任何新模型；
- hidden correctness 不参与 susceptibility label。

## 主要结果变量

对每个 task：

greedy_divergence_rate
= 12 个 matched perturbation arms 中，
  greedy raw completion 与 SFT greedy 不同的 arm 数 / 12。

12 个 arms：

- scale 0.25 × seeds 101/202/303；
- scale 0.5 × seeds 101/202/303；
- scale 1 × seeds 101/202/303；
- scale 2 × seeds 101/202/303。

同时记录：

- divergence rate by scale；
- total divergent arms；
- SFT greedy output length。

## SFT trajectory margin profile

在 SFT greedy 自己生成的 token path 上，对每个生成位置计算：

- top1 probability；
- top2 probability；
- probability margin = top1 - top2；
- logit margin = top1_logit - top2_logit。

主要窗口：

- 前 min(128, completion_length) 个生成 token。

同时保留 full trajectory summary。

每个 task 记录：

- mean / median / min margin；
- p10 / p25 margin；
- fraction(margin ≤ 1e-6)；
- fraction(margin ≤ 0.01)；
- fraction(margin ≤ 0.05)；
- fraction(margin ≤ 0.10)；
- token count；
- greedy token 与模型 top1 一致率。

## 预注册主要假设

主要 predictor：

first128_probability_margin_fraction_le_0_05

主要 outcome：

greedy_divergence_rate

主要检验：

- Spearman correlation；
- 20,000 次 task-level bootstrap；
- seed=42。

支持 susceptibility 假设：

> Spearman rho > 0 且 95% bootstrap CI 完全大于 0。

## 长度控制

为了排除“序列越长，遇到分叉点机会越多”这一替代解释，额外拟合：

divergence_rate
~ standardized(low_margin_fraction_le_0_05)
+ standardized(log1p(token_count))

对 low-margin coefficient 做 20,000 次 task bootstrap。

若 low-margin coefficient 的 95% CI 仍大于 0，则说明信号不能仅由 sequence length 解释。

## 次要分析

1. p10 probability margin 与 divergence rate 的 Spearman（预期负相关）；
2. median / min margin；
3. tie fraction；
4. predictor top quartile vs bottom quartile 的平均 divergence rate；
5. 各 scale 分别的 susceptibility；
6. task correctness / output length 仅作为诊断，不参与主要结论。

## Phase B（只有 Phase A 完成后决定）

若 SFT margin profile 对 susceptibility 有预测力，再比较：

- Base greedy margin profile；
- SFT greedy margin profile；

回答：

> SFT 是否系统性把任务轨迹推向更多 low-margin 决策区域？

该比较在 Phase A 结果出来前不提前解释。

## 记录

- experiments/EXP-004G-trajectory-susceptibility.md
- results/exp004g/
- docs/daily/2026-10-08.md

失败、性能问题、重新运行和负结果均记录。


# Phase A 正式结果：SFT margin profile 可以预测 perturbation susceptibility

## 复现检查

SFT greedy trajectory margin profiler 重新生成 90 个 validation task：

- 与 EXP-004F 正式 SFT greedy raw completion：**90 / 90 完全一致**；
- 因此 margin profile 与 12-arm susceptibility label 严格对齐。

12 个 matched-perturbation greedy arm 的平均 task divergence rate：**25.37%**，task 范围 0%–100%。

## 预注册主要检验

主要 predictor：前 min(128, completion length) token 中 probability top1-top2 margin ≤ 0.05 的比例。

主要 outcome：12-arm greedy divergence rate。

Spearman：

- ρ = **0.3166**；
- 95% bootstrap CI = **[0.1115, 0.4991]**；
- P(ρ>0) = 0.9989。

预注册主要假设通过。

## 长度控制

回归：

divergence_rate ~ standardized(low-margin fraction) + standardized(log1p(token count))

结果：

- low-margin coefficient = **+0.1033**；
- 95% bootstrap CI = **[+0.0373,+0.1637]**；
- P(coef>0) = 0.9994；
- length coefficient = +0.0631。

因此 predictor 不能简单解释为“输出越长，遇到分叉机会越多”。

## Quartile 对照

按主要 predictor 排序：

- bottom quartile 平均 low-margin fraction = 0；
- bottom quartile 平均 divergence rate = **17.05%**；
- top quartile 平均 low-margin fraction = 5.87%；
- top quartile 平均 divergence rate = **39.39%**；
- top - bottom = **+22.35 个百分点**。

## 次要 margin 指标

| Predictor | Spearman ρ | 95% CI |
|---|---:|---:|
| margin≤0.01 fraction | +0.2866 | [+0.0746,+0.4786] |
| margin≤0.05 fraction | **+0.3166** | **[+0.1115,+0.4991]** |
| margin≤0.10 fraction | +0.3430 | [+0.1415,+0.5255] |
| tie fraction | +0.2849 | [+0.0727,+0.4781] |
| p10 margin | -0.0918 | [-0.3159,+0.1361] |
| median margin | +0.0654 | [-0.1542,+0.2785] |
| min margin | **-0.2796** | **[-0.4594,-0.0739]** |

“低 margin 出现频率”比单个 p10/median 更有预测力；min margin 越小则 susceptibility 越高。

## 跨 perturbation scale 复现

主要 predictor 对每个 scale 的 divergence rate：

| Scale | ρ | 95% CI |
|---|---:|---:|
| 0.25× | +0.2770 | [+0.0672,+0.4662] |
| 0.5× | +0.2943 | [+0.0813,+0.4904] |
| 1× | **+0.3600** | **[+0.1575,+0.5428]** |
| 2× | +0.2955 | [+0.0936,+0.4809] |

四个尺度全部正相关且 bootstrap CI >0。

## Phase A 结论

> SFT greedy 轨迹中 low-margin token 的密度，可以在不知道 perturbation 结果的情况下预测 task-level trajectory susceptibility。

这把 EXP-004F 的机制从“事后解释首次分叉”推进到了“事前预测哪些任务更脆弱”。

# Phase B 预注册：Base vs SFT margin profile

Phase A 已满足预注册成功标准，因此进入 Phase B。

问题：

> SFT 是否系统性改变 task 的 low-margin exposure，使模型更靠近局部自回归决策边界？

设计：

- 同 90 个 MBPP validation task；
- Base 使用自己的 deterministic greedy trajectory；
- SFT 使用自己的 deterministic greedy trajectory；
- 两者都计算前128 token probability margin profile；
- 不要求 token-level 路径对齐，只做 task-level paired profile 比较。

主要量：

- first128 fraction(margin≤0.05)。

主要比较：

- paired SFT - Base task difference；
- 20,000 task bootstrap；
- 双向解释：CI>0 支持 SFT 增加 low-margin exposure；CI<0 则反驳这一简单解释。

次要：

- tie fraction；
- margin≤0.01 / ≤0.10；
- min / p10 margin；
- completion length；
- SFT-Base low-margin delta 与 SFT perturbation susceptibility 的 Spearman。


# Phase B 正式结果：SFT 并没有全局增加 low-margin exposure

预注册主要量：前 min(128, completion length) token 中 probability margin ≤0.05 的比例。

Base：

- mean low-margin fraction = 3.5765%

SFT：

- mean low-margin fraction = 2.2918%

SFT - Base：

- mean delta = **-1.2847 个百分点**；
- 95% bootstrap CI = **[-1.9123,-0.6441]** 个百分点；
- 30 个 task 增加；
- 57 个 task 减少；
- 3 个不变。

因此预注册假设“SFT 系统性增加 low-margin exposure”被明确反驳。

## 其它 margin 指标

### margin≤0.10

- Base mean = 6.4922%；
- SFT mean = 3.6549%；
- delta = **-2.8373 pp**；
- 95% CI = **[-3.8344,-1.8483] pp**。

### min margin

- Base mean = 0.00525；
- SFT mean = 0.05116；
- delta = **+0.04591**；
- 95% CI = **[+0.02931,+0.06566]**。

### p10 margin

- Base mean = 0.2367；
- SFT mean = 0.4083；
- delta = **+0.1716**；
- 95% CI = **[+0.1214,+0.2217]**。

### median margin

- Base mean = 0.9306；
- SFT mean = 0.9537；
- delta = +0.0230；
- 95% CI = [+0.0028,+0.0446]。

整体看，SFT greedy 轨迹反而更高 margin、更确定。

## 输出长度变化

Base greedy completion mean token count：464.6。

SFT：69.67。

SFT - Base：**-394.93 token**；95% CI [-421.87,-365.28]。

Base 大量任务跑到 512 token 上限，而 SFT 输出明显更短、更结构化。

因此 Base-vs-SFT self-trajectory profile 不能简单解释成“同一轨迹上的 margin 变高/变低”；SFT 已经改变了轨迹本身。

## margin shift 与 susceptibility

task-level：

SFT-Base 的 low-margin-fraction delta vs SFT perturbation susceptibility：

- Spearman rho = +0.2174；
- 95% CI = [-0.0050,+0.4223]；
- P(rho>0)=0.9724。

点估计为正，但区间跨 0，当前不能确认“被 SFT 推向更多 low-margin 的 task 就更 susceptible”。

## Phase B 结论

当前证据否定一个简单故事：

> SFT 不是通过全局降低 token decision margin 来制造脆弱性。

更符合数据的是：

1. SFT 整体使输出大幅变短；
2. SFT 自己的 greedy 轨迹整体 margin 反而更高；
3. 但在 SFT 轨迹内部，少量 low-margin / tie token 的局部密度仍然能强预测 perturbation susceptibility；
4. 因此 susceptibility 更像由**局部 bottleneck 决策点**驱动，而不是由全局 margin 水平决定。

这与 EXP-004F 的首次分叉分析一致：真正触发级联分叉的是局部 near-tie token，而不是 prompt 起点或整条轨迹都处于低 margin 状态。

## 下一问题

不继续尝试证明“SFT 全局 lowering margin”。该假设已被数据反驳。

更有价值的下一步是直接研究局部 bottleneck：

- 是否可以仅根据 SFT 的 margin profile 选出少量高 susceptibility token/task；
- 这些 bottleneck token 是否在不同 perturbation seed/scale 间重复出现；
- 对 bottleneck 附近做极小 logit/margin 干预，是否能改变 trajectory susceptibility；
- 从“相关预测”推进到“局部决策边界的因果干预”。
