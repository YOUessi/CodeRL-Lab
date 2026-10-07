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
