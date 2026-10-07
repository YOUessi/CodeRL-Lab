# EXP-004I：Local Bottleneck Correctness Causality

## 状态

- 分支：exp/bottleneck-correctness-causality
- 设计：已预注册
- 不重新生成 GPU trajectories
- correctness evaluator：待运行
- task-cluster bootstrap：待运行

## 研究动机

EXP-004H 已经证明 reference-defined local low-margin bottleneck 对 greedy trajectory divergence 具有强因果作用：

- low-stabilize 相对 baseline exact-match +13.16 pp；
- high-margin stabilize control = 0；
- low-destabilize 相对 baseline exact-match -53.51 pp；
- low-stabilize rescue 30/78 baseline-divergent pair；
- low-destabilize 新诱发分叉 122/150 baseline-matching pair。

但是：

> 轨迹更像 SFT reference，不等于代码更正确。

如果 SFT reference 本身错误，stabilize 甚至可能把 candidate 锁回错误轨迹。

EXP-004I 因此只回答 correctness causality。

## 固定输入

不重新做模型生成。

直接复用 EXP-004H 四个正式 arm：

- scale025_seed101
- scale050_seed202
- scale100_seed202
- scale200_seed303

每个 arm 的：

- baseline
- low_stabilize
- low_destabilize
- high_stabilize

完整 raw completion 已固定在 artifacts/exp004h/arms/。

Reference SFT 与 candidate baseline correctness 直接复用 EXP-004F 的 hidden-test 评测。

只有三类 intervention output 重新运行 hidden tests：

- low_stabilize
- low_destabilize
- high_stabilize

## Target 选择与泄漏约束

eligible task 与 intervention target 完全沿用 EXP-004H：

- reference SFT first128；
- first margin <= 0.05；
- hidden correctness 不参与 target 选择。

因此 EXP-004I 的 hidden tests 只用于后验 outcome evaluation，不影响干预位置。

## Primary outcomes

在 57 个 eligible task × 4 perturbation arms 上，以 task 为 bootstrap cluster。

### 1. low-stabilize correctness - baseline

如果 95% CI > 0：

说明局部稳定 bottleneck 不仅保持轨迹，也提高正确性。

如果 CI 跨 0：

说明 trajectory causality 不自动转化为 correctness causality。

### 2. low-stabilize - high-margin control

控制相同 +0.25 bias，但位置不是 bottleneck。

用于判断 correctness 变化是否具有 low-margin position specificity。

### 3. low-destabilize - baseline

判断反方向局部干预是否破坏正确性，或是否偶尔解锁新的正确轨迹。

## Transition counts

每个 condition 都报告：

- wrong -> correct
- correct -> wrong
- correct -> correct
- wrong -> wrong

不能只报平均正确率。

## Reference-correctness stratification

按 SFT reference hidden correctness 分层：

- reference correct
- reference wrong

重点判断：

### reference correct

low-stabilize 是否更容易 rescue correctness。

### reference wrong

low-stabilize 是否会 lock-in 错误；
low-destabilize 是否反而有机会 unlock 正确轨迹。

## Trajectory rescue vs correctness rescue

对 EXP-004H 中：

baseline != reference
且
low_stabilize == reference

的 trajectory-rescue pair，额外报告：

- reference correct / wrong；
- baseline correct / wrong；
- low-stabilize correct / wrong；
- correctness rescue 数；
- correctness harm 数。

这样将“轨迹 rescue”与“能力 rescue”严格分开。

## 统计

- task-cluster bootstrap；
- 20,000 iterations；
- seed=42；
- cluster=task_id；
- 四个 perturbation arm 不作为独立 task。

## 结论边界

本实验回答：

> local low-margin intervention 是否对 hidden-test correctness 有因果作用。

它不训练新模型，也不证明一种可部署解码算法。

如果 trajectory effect 强、correctness effect 弱：

说明 bottleneck 主要控制 trajectory identity，而非任务能力。

如果 correctness effect 同样稳定：

才值得进一步研究 bottleneck-aware decoding / training。
