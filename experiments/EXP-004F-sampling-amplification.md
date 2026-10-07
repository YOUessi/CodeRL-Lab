# EXP-004F：随机采样放大效应 / Logit 敏感性

## 状态

- 分支：exp/sampling-amplification
- greedy generation mode：已实现
- prompt-end logit sensitivity：已实现
- shared reference-logit cache：已实现
- CPU tests：待跑
- 3-task smoke：待跑
- 90-task 正式 greedy：待跑
- 90-task prompt-logit analysis：待跑

## 研究动机

EXP-004D/E 已经证明：

- 对 SFT LoRA adapter 施加极小 matched-norm 随机参数扰动；
- 不经过 DPO / optimizer / preference data；
- 也能稳定恢复高 k 覆盖；
- fixed-seed stochastic completion 约 29%–31% 会发生变化；
- 但从 0.25× 到 2×，行为漂移并不随参数距离单调增加。

EXP-004E 剂量结果：

| Scale | 平均 ΔPass@4 | 平均 ΔPass@8 | 平均 ΔPass@16 |
|---|---:|---:|---:|
| 0.25× | +1.45 pp | +2.17 pp | +3.33 pp |
| 0.5× | +2.41 pp | +3.61 pp | +5.19 pp |
| 1× | +1.20 pp | +2.05 pp | +3.70 pp |
| 2× | +1.98 pp | +3.26 pp | +4.81 pp |

同时 fixed-seed exact completion match 在所有非零 scale 都约 68%–70%。

因此下一问题不是“参数扰动有没有作用”，而是：

> 参数扰动究竟显著改变了模型的条件分布，还是微小 logit 改变被随机采样和 autoregressive feedback 放大成大规模轨迹分叉？

## Phase A：确定性 Greedy 稳定性

固定：

- Qwen3-1.7B-Base；
- 同 SFT adapter；
- 同 matched-norm perturbation arms；
- MBPP validation 90 tasks；
- greedy decoding；
- num_samples=1；
- 不使用随机采样。

比较：

- exact greedy completion match；
- changed tasks；
- hidden correctness；
- greedy solved tasks；
- 不同 scale / seed 的稳定性。

### 预注册解释

强 sampling-amplification 证据：

- stochastic exact match 约 70%，但 greedy exact match 明显更高，例如 >90%；
- greedy task correctness 基本稳定。

反之，若 greedy 也只有约 70% 一致，则说明 tiny perturbation 已明显改变 deterministic model behavior。

## Phase B：Prompt 末端 next-token 分布

对同一 prompt，在生成第一个 completion token 前比较：

- KL(reference || candidate)；
- total variation distance；
- centered logit RMS delta；
- top-1 token agreement；
- top-5 token Jaccard；
- top1-top2 probability margin；
- candidate 对 reference top1 token 的概率。

SFT prompt-end logits 对 90 task 只计算一次并缓存，本地 artifact 不进入 Git。

### 预注册解释

更支持 sampling amplification：

- top1 agreement 很高；
- KL / TV 很小；
- 但 stochastic trajectory drift 已约 30%。

更支持模型条件分布显著变化：

- top1 大量翻转；
- KL / TV 已明显；
- greedy completion 同样大量变化。

## Phase C（仅在 A/B 后决定）：Temperature 扫描

若 A/B 支持 sampling amplification，再固定相同 adapter arms，比较：

- temperature 0.2；
- 0.5；
- 0.8；
- 1.0；

下的 fixed-seed behavior drift。

预期：

如果采样是主要放大器，temperature 越低，行为漂移应下降；greedy 应最低。

## 正式 arms

保持 EXP-004E 的 12 个 matched perturbation：

- scales：0.25 / 0.5 / 1 / 2；
- seeds：101 / 202 / 303。

SFT 为 reference。

## 记录要求

- experiments/EXP-004F-sampling-amplification.md
- results/exp004f/
- docs/daily/2026-10-07.md

必须记录：

- 工程失败；
- 代码修复；
- greedy 与 stochastic 差异；
- logit 指标；
- 不支持预期的负结果；
- 后续是否进入 temperature sweep。


# Phase A/B 正式结果

## Prompt 起点分布

12 个 matched-perturbation arm（scale 0.25/0.5/1/2 × seed 101/202/303）全部完成 90-task prompt-end logit 分析。

共同结果：

- prompt top-1 agreement：12/12 arm = **100%**；
- mean KL(reference||candidate)：约 **0.00096–0.00146**；
- mean TV：约 **0.0068–0.0106**；
- top-5 Jaccard：约 **0.956–0.981**。

因此 tiny perturbation 并没有在 prompt 起点大规模改变最可能 token。

## Greedy 完整轨迹

尽管 prompt 起点 top-1 100% 一致，完整 deterministic greedy completion 仍大量分叉。

三 seed 平均：

| Scale | stochastic changed | greedy changed | stochastic-extra gap |
|---|---:|---:|---:|
| 0.25× | 31.46% | 27.41% | +4.05 pp |
| 0.5× | 30.93% | 23.33% | +7.59 pp |
| 1× | 30.16% | 21.11% | +9.05 pp |
| 2× | 30.37% | 26.67% | +3.70 pp |

因此：

> tiny perturbation 已能通过 deterministic autoregressive feedback 造成约 21%–27% 的 task-level greedy trajectory divergence；随机采样通常在此基础上再增加约 4–9 个百分点的额外分叉。

这否定了“只有随机采样才会放大扰动”的初始强假设。

## 首次 greedy 分叉机制

预注册代表 arms：

- 0.25× seed101；
- 0.5× seed202；
- 1× seed202；
- 2× seed303。

结果：

| Arm | divergent tasks | 首次分叉中位 token | ≤50 token | KL median | TV median | ref top1 margin median |
|---|---:|---:|---:|---:|---:|---:|
| 0.25× s101 | 32 | 17.5 | 87.5% | 0.00544 | 0.04356 | 0 |
| 0.5× s202 | 19 | 17 | 84.2% | 0.00124 | 0.00892 | 0.0667 |
| 1× s202 | 21 | 17 | 81.0% | 0.00111 | 0.00937 | 0 |
| 2× s303 | 26 | 13.5 | 92.3% | 0.00251 | 0.03216 | 0 |

四臂共 98 个首次分叉事件：

- reference top1 margin = 0：**56.12%**；
- reference margin ≤ 0.05：**60.20%**；
- reference margin ≤ 0.10：**78.57%**；
- KL ≤ 1e-3：**36.73%**；
- KL ≤ 1e-2：**94.90%**；
- TV ≤ 1%：**40.82%**；
- TV ≤ 5%：**62.24%**；
- 首次分叉 ≤10 token：29.59%；
- 首次分叉 ≤50 token：**86.73%**。

甚至存在 KL 约 1e-6、TV 约 1e-5 的首次分叉事件。

## Phase A/B 机制结论

当前更符合：

tiny local parameter perturbation
→ prompt 起点分布只有微小变化，top1 不变
→ 自回归生成过程中不断遇到低 margin / tie 决策点
→ 某个 token 的 argmax 被极小 logit 变化翻转
→ 新 prefix 反馈到后续条件分布
→ deterministic greedy trajectory 级联分叉
→ stochastic sampling 再叠加额外分叉

因此机制应称为：

> **deterministic autoregressive amplification + additional stochastic amplification**

而不是单纯 sampling amplification。

# Phase C 预注册：Temperature 扫描

Phase A/B 已证明存在额外 stochastic amplification，因此进入温度扫描。

固定代表 arms：

- 0.25× seed101；
- 0.5× seed202；
- 1× seed202；
- 2× seed303。

温度：

- 0.2；
- 0.5；
- 0.8；
- 1.0。

每个 temperature：

- SFT reference 只生成一次；
- 每个 arm 90 tasks × 4 samples；
- generation seed=42；
- top_p=0.95；
- max_new_tokens=512；
- 与相同 temperature 的 SFT 按 task_id/sample_id 对齐；
- 比较 exact completion match / changed fraction；
- 同时报告 hidden correctness 与 Pass@1/4。

greedy Phase A 作为 temperature→0 的确定性参照。

预注册判断：

1. 若 temperature 越高，candidate-vs-SFT changed fraction 系统上升，则 stochastic sampling amplification 得到直接支持；
2. 若 0.2 已接近 0.8/1.0，则 stochastic temperature 不是主要剩余因素，自回归决策边界占主导；
3. 不根据结果调整代表 arm 或 temperature。
