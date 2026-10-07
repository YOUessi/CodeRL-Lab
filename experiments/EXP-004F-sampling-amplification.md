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
