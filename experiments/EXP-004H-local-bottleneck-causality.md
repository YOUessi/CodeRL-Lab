# EXP-004H：Local Bottleneck Causality / 低 Margin 决策点因果干预

## 状态

- 分支：exp/local-bottleneck-causality
- 设计：已预注册
- intervention runner：待实现
- CPU tests：待实现
- GPU smoke：待运行
- 正式 4-arm × 90-task 干预：待运行
- 统一统计：待运行

## 研究动机

EXP-004F 已经证明：

- prompt 起点 top1 在 12/12 matched perturbation arm 中保持 100% 一致；
- greedy 轨迹仍有约 20%–34% task 分叉；
- 首次分叉经常发生在 low-margin / near-tie token。

EXP-004G Phase A 进一步证明：

- SFT greedy 轨迹中 low-margin token 密度可以事前预测 perturbation susceptibility；
- margin≤0.05 fraction vs 12-arm divergence rate：Spearman rho=0.3166；
- 95% CI [0.1115,0.4991]；
- 控制输出长度后信号仍成立。

但 Phase B 否定了简单的全局解释：

- Base margin≤0.05 mean = 3.58%；
- SFT = 2.29%；
- SFT-Base = -1.28 pp，95% CI [-1.91,-0.64]；
- SFT greedy 输出也从 Base 464.6 token 缩短到 69.7 token。

因此当前机制不是“SFT 全局 lowering margin”，而更像：

> SFT 轨迹整体更短、更高 margin，但其中稀疏 local near-tie bottleneck 决定了 perturbation susceptibility。

EXP-004H 的目标是从相关预测推进到局部因果干预。

## 固定模型与任务

Reference policy：

- Qwen3-1.7B-Base
- EXP-006A SFT adapter

Candidate perturbation arms 固定复用 EXP-004F 预注册代表 arm：

1. 0.25× seed101
2. 0.5× seed202
3. 1× seed202
4. 2× seed303

任务：

- MBPP validation 90 tasks
- deterministic greedy
- max_new_tokens=512

不训练任何新模型。

## Reference bottleneck 定义

对每个 task，重新生成 SFT greedy reference trajectory，并在前：

min(128, completion length)

个生成 token 上计算 reference probability top1-top2 margin。

### 主目标位置

选择**第一个**：

margin ≤ 0.05

的生成位置。

这样目标完全由 reference SFT 决定，不使用：

- candidate perturbation 结果；
- first divergence 结果；
- hidden correctness。

如果一个 task 在前128 token 中没有 margin≤0.05，则记为 ineligible，不进入主要 intervention effect 估计，但仍保留诊断信息。

## 三种干预

固定 logit bias 绝对值：

0.25

### A. low-margin stabilize

在主目标位置：

- candidate prefix 必须在该位置之前仍与 SFT reference prefix 完全一致；
- 给 SFT reference token 的 candidate logit 加 +0.25；
- 然后继续 greedy，不再做额外干预。

目的：

> 提高 near-tie reference decision 的局部稳定性，看是否能阻止后续 trajectory divergence。

### B. low-margin destabilize

同一主目标位置：

- 给 SFT reference token logit 加 -0.25；
- 其它完全相同。

目的：

> 检查反方向干预是否更容易诱发或提前 trajectory divergence。

### C. high-margin stabilize control

为每个 eligible task，在前128 token 中找：

margin ≥ 0.20

且与主目标 token index 时间距离最近的位置。

在该位置给对应 SFT reference token +0.25。

因此：

- 干预次数相同；
- bias 幅度相同；
- 都是 reference token stabilization；
- 主要差别是干预位置是否为 low-margin bottleneck。

## Prefix 对齐原则

干预只在 candidate 在目标位置前仍与 SFT reference prefix 完全一致时执行。

如果 candidate 在目标位置前已经分叉：

- 不强行把 prefix 改回 reference；
- intervention_applied=false；
- 保留该 task-arm-condition 作为诊断。

这样不会把一个已经不同的 trajectory 人工 teleport 回 reference path。

## 既有 no-intervention baseline

不重新生成 baseline。

直接复用 EXP-004F 对应四个 candidate greedy arm：

- scale025_seed101
- scale050_seed202
- scale100_seed202
- scale200_seed303

以及同一 SFT greedy reference。

## 主要结果

### 1. Exact greedy trajectory match

对 eligible task × arm：

- baseline exact match to SFT；
- low-stabilize exact match；
- low-destabilize exact match；
- high-stabilize exact match。

### 2. Rescue

对于 baseline 已分叉的 task-arm：

- low-stabilize 是否恢复成与 SFT 完整 raw completion 一致；
- high-stabilize 是否恢复。

主要关注：

low-stabilize rescue > high-stabilize rescue

### 3. Induced divergence

对于 baseline 原本与 SFT 完全一致的 task-arm：

- low-destabilize 是否诱发新分叉。

### 4. First divergence position

若仍分叉：

- intervention 后首次 divergence index 是否后移/前移。

## 主要预注册判断

支持 local bottleneck causal hypothesis：

1. low-stabilize 相对 baseline 提高 exact-match / rescue；
2. 该改善大于 high-margin stabilize control；
3. low-destabilize 相对 baseline 增加 divergence；
4. 方向在多个 candidate arms 中一致。

统计：

- task-cluster bootstrap；
- 20,000 iterations；
- seed=42；
- task 为 bootstrap cluster，避免把四个 perturbation arm 当成独立 task。

## Secondary / diagnostic

- eligible task 数；
- first bottleneck position 分布；
- bottleneck margin 分布；
- intervention_applied fraction；
- baseline divergence 是否发生在 bottleneck 前/后；
- hidden correctness 只做后验诊断，不参与 target 或主要因果定义。

## 工程约束

- GitHub 是代码/记录唯一事实源；
- Tang 只做 GPU 推理；
- GPU arm 串行，不并发；
- 任何 smoke / 失败 / 重跑写入 docs/daily/2026-10-08.md；
- hidden tests 不参与 intervention target 选择。

## 结论边界

即便 low-margin stabilize 能显著减少 divergence，也只能说明：

> 在当前 Qwen3-1.7B SFT + MBPP greedy 轨迹上，局部 near-tie decision boundary 对 tiny-perturbation-induced trajectory divergence 具有因果作用。

不能直接外推到：

- 所有模型；
- 所有任务；
- 所有 decoding regime；
- 模型“能力”本体。



## 4-task GPU smoke

- CPU tests：133/133 passed；
- SFT reference reproduction：4/4；
- candidate baseline reproduction：4/4；
- eligible tasks：2/4。

关键 smoke：mbpp_validation_0512

- reference first low-margin position：token 17；
- margin = 0；
- baseline candidate first divergence：17；
- low-stabilize +0.25 在 token17 成功应用；
- 首次 divergence 被推迟到 token21；
- high-margin control（token16，margin≈0.997）没有改变 baseline divergence；
- low-destabilize 也没有产生额外延迟。

另一个 eligible task 0511 的 baseline 在 token65 已提前分叉，而第一个 low-margin target 在 token101，因此 intervention_applied=false；按预注册规则不强制把轨迹拉回 reference。

Smoke 证明：干预位置与 prefix-alignment 逻辑正确，允许进入正式 4-arm × 90-task 因果实验。

正式分析除 exact rescue 外，增加 first-divergence survival index：若完整匹配则记为 reference token length，否则使用首次 divergence index；报告 stabilize/destabilize 相对 baseline 推迟或提前的 token 数。
