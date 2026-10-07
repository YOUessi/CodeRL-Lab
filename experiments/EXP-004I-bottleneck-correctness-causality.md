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


# 正式 correctness 因果结果

EXP-004H 的四个 intervention arm 已全部使用同一 Docker hidden tests 做 correctness 重判。

主要分析：57 eligible tasks × 4 perturbation arms = 228 task-arm pairs。

## 总体 hidden correctness

| 条件 | hidden correct fraction | 相对 baseline |
|---|---:|---:|
| baseline | 34.21% | — |
| low-stabilize | 33.33% | -0.88 pp |
| low-destabilize | **38.60%** | +4.39 pp |
| high-margin stabilize | 34.21% | 0 |

### low-stabilize - baseline

- delta：-0.88 pp；
- 95% CI：[-2.63, 0]；
- wrong→correct：0；
- correct→wrong：2。

因此：

> EXP-004H 中显著的 trajectory rescue 并没有转化为 correctness rescue。

### low-destabilize - baseline

- delta：+4.39 pp；
- 95% CI：[-1.32,+11.40]；
- wrong→correct：14；
- correct→wrong：4。

总体点估计偏正，但总体区间仍跨0。

### high-margin stabilize - baseline

- delta：0；
- 所有 correctness transition 均为0。

## 按 SFT reference correctness 分层

这是本实验最关键的结果。

### SFT reference 正确

19 tasks / 76 task-arm pairs：

- baseline correct：98.68%；
- low-stabilize：98.68%；
- low-destabilize：93.42%。

low-stabilize - baseline：0。

low-destabilize - baseline：

- delta：-5.26 pp；
- 95% CI：[-14.47,0]。

因此 reference 本身正确时，打破 bottleneck 没有收益，反而有伤害倾向。

### SFT reference 错误

38 tasks / 152 task-arm pairs：

- baseline correct：1.97%；
- low-stabilize：0.66%；
- low-destabilize：**11.18%**。

low-stabilize - baseline：

- delta：-1.32 pp；
- 95% CI：[-3.95,0]。

low-destabilize - baseline：

- delta：**+9.21 pp**；
- 95% CI：**[+1.97,+18.42]**；
- P(delta>0)=0.9957。

因此：

> 当 SFT reference trajectory 本身错误时，在其预注册 local low-margin bottleneck 上做反向干预，能够显著增加 hidden-test 正确轨迹出现概率。

## Trajectory rescue 与 correctness rescue 的分离

EXP-004H 中 baseline!=reference 且 low-stabilize==reference 的完整 trajectory rescue pair 共30个：

- reference correct：12；
- reference wrong：18；
- correctness rescue：**0**；
- correctness harm：**2**。

这直接证明：

> **trajectory rescue 不等于 capability/correctness rescue。**

让 candidate 更像 SFT reference 甚至可能把原本正确但不同的 candidate 拉回错误 SFT 轨迹。

## Destabilize-induced divergence 的 correctness 结果

EXP-004H 中 baseline 原本与 SFT 完全一致、但 low-destabilize 新诱发分叉的 pair 共122：

- wrong→correct：10；
- correct→wrong：1。

因此“离开 SFT trajectory”并不天然是坏事；当 SFT reference 错时，它反而可能是解锁正确轨迹的必要步骤。

## EXP-004I 最终机制结论

EXP-004H 证明 low-margin bottleneck 控制 trajectory identity。

EXP-004I 进一步证明：

1. stabilize bottleneck 能让 candidate 更像 SFT，但不提高 correctness；
2. 当 SFT reference 正确时，保持其局部决策通常是安全的；
3. 当 SFT reference 错误时，destabilize 同一个 bottleneck 可以显著提高 correctness；
4. 因此 local bottleneck 更像是**轨迹分叉闸门**，而不是“正确答案方向”的标记。

更准确的机制是：

```text
reference SFT 正确
→ low-margin bottleneck 稳定化通常维持正确轨迹

reference SFT 错误
→ low-margin bottleneck 稳定化会锁住错误轨迹
→ destabilize 提供离开错误 attractor / trajectory 的机会
→ 一部分 alternative trajectory 变为正确
```

## 下一问题：可观测 gating

hidden correctness 不能用于部署时决定 stabilize / destabilize。

因此下一步不直接做新的 GPU intervention，而先检查：

> **SFT reference 的 public-test 结果能否作为 hidden correctness 的可观测 proxy，从而决定什么时候应该 destabilize。**

如果 public-fail subset 能稳定复现 reference-wrong subset 的正向 destabilize effect，才值得进入 verifier-gated bottleneck decoding。


# Public-test gating audit

hidden correctness 不能用于部署时决定 intervention sign，因此进一步检查 SFT reference 的 public-test 结果能否作为可观测 proxy。

57 个 eligible task：

- public-fail：36；
- hidden-wrong：38；
- public-fail ∩ hidden-wrong：34。

因此：

- public-fail 对 hidden-wrong 的 precision：**94.44%**；
- recall：**89.47%**。

## public-fail stratum

36 tasks / 144 task-arm pairs：

- baseline correctness：6.25%；
- low-destabilize：**14.58%**；
- delta：**+8.33 pp**；
- 95% CI：**[+0.69,+18.06] pp**；
- wrong→correct：12；
- correct→wrong：0。

low-stabilize 在该 stratum 为 0 改善。

## public-pass stratum

21 tasks / 84 pairs：

- baseline correctness：82.14%；
- low-destabilize：79.76%；
- delta：-2.38 pp；
- 95% CI：[-11.90,+5.95] pp。

因此 public tests 能有效区分“更适合打破 SFT trajectory”的任务。

# 离线 public-gated policy

规则：

```text
if SFT reference public tests fail:
    choose low-destabilize candidate
else:
    keep baseline candidate
```

在 228 eligible task-arm pairs 上：

- baseline correctness：34.21%；
- gated correctness：**39.47%**；
- delta：**+5.26 pp**；
- 95% CI：**[+0.44,+11.40] pp**；
- P(delta>0)=0.9842；
- wrong→correct：12；
- correct→wrong：0。

这说明 EXP-004I 的 oracle hidden-correctness 分层可以被一个训练/推理时可见的 public-test gate 近似实现。

## EXP-004I 最终结论

1. low-margin bottleneck 对 trajectory identity 有强因果作用；
2. stabilize reference trajectory 本身不提高 correctness；
3. 当 SFT reference 错误时，destabilize 能显著解锁正确 alternative trajectory；
4. public-test failure 对 hidden-wrong 有高 precision / recall；
5. 基于 public-test 的 offline gate 得到正向 task-cluster bootstrap 区间。

因此下一步进入更严格的 SFT-only 两阶段验证：

> **先运行 SFT greedy + public tests；仅在 public fail 且存在 low-margin bottleneck 时，对 SFT 自己的 bottleneck 做一次 destabilize 重生成。**

这将移除“candidate perturbation adapter”这个中间变量，直接测试 verifier-gated local escape 是否具有可部署价值。
