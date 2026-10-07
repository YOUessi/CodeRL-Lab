# EXP-004E：局部参数扰动幅度剂量—响应

## 状态

- 分支：exp/perturbation-dose-response
- scale 参数：已实现
- scale / direction 单元测试：已实现
- 正式实验：待运行

## 研究动机

EXP-004D 已经证明：

- 与 No-op DPO 同量级；
- 方向随机且彼此近乎正交；
- 不经过 DPO / preference / optimizer；

的局部 LoRA 参数扰动，也能稳定提高高 k 覆盖。

三 seed 平均：

- Pass@4 +1.20 pp；
- Pass@8 +2.05 pp；
- Pass@16 +3.70 pp；
- solved@16 +3.33；
- success HHI 下降；
- effective task count 上升。

因此下一问题不再是“DPO 是否必要”，而是：

> 覆盖恢复与扰动幅度之间是什么关系？

## 预注册实验

### 固定基准

Base adapter：

EXP-006A 1.7B SFT

Reference norm source：

EXP-004C No-op dropout=0.05 - SFT

Reference global delta L2：

约 0.0062008071。

### 幅度

统一相对 No-op05 per-tensor delta norm 的 scale：

- 0×：原始 SFT，不生成新 adapter；
- 0.25×；
- 0.5×；
- 1×：直接复用 EXP-004D；
- 2×。

### 随机方向

固定三条已经在 EXP-004D 使用的随机方向：

- seed 101；
- seed 202；
- seed 303。

同一个 seed 在不同 scale 下必须使用完全相同的随机方向，只改变 norm。

### 新增计算

需要新跑：

- 0.25× × 3 seeds；
- 0.5× × 3 seeds；
- 2× × 3 seeds。

共 9 个新 arm。

1× 的 3 个 arm 复用 EXP-004D；
0× 直接使用 SFT。

## 评测

每个 arm：

- MBPP validation 90 tasks；
- n=16；
- temperature=0.8；
- top_p=0.95；
- generation seed=42；
- Docker hidden tests；
- Pass@1/4/8/16；
- solved@16；
- hidden mean；
- fixed-seed exact completion match；
- success-mass HHI；
- effective task count；
- exact-output diversity。

## 预注册解释

### 支持连续剂量—响应

如果从 0 → 0.25 → 0.5 → 1：

- 高 k 覆盖逐步上升；
- HHI 逐步下降；
- fixed-seed 行为漂移逐步增加；

说明去集中与局部参数位移幅度存在连续关系。

### 支持“适量扰动最佳”

如果：

- 0.25/0.5/1 提升；
- 2× 开始下降或破坏 hidden correctness；

说明存在局部最佳扰动半径。

### 支持“阈值/不连续敏感性”

如果很小 scale 已产生接近 1× 的行为改变和覆盖恢复，且后续幅度增加不再明显增益，则说明 SFT 位于局部高敏感区域，存在近似阈值效应。

### 反驳局部敏感性解释

如果不同 scale / seed 结果高度随机，无稳定幅度关系，或者多数随机扰动破坏能力，则 EXP-004D 的 3-seed 正向结果可能是有限 seed 偶然。

## 记录要求

- experiments/EXP-004E-perturbation-dose-response.md：实验纵向记录；
- results/exp004e/：机器可读结果；
- docs/daily/2026-10-07.md：今日研发流水；
- 所有失败、重跑、脚本错误、环境问题均保留。

# 正式剂量—响应结果

固定 3 个随机方向 seed=101/202/303，仅改变每 tensor 扰动幅度：0.25× / 0.5× / 1× / 2×。

## 三 seed 平均增益（相对 SFT）

| Scale | ΔPass@1 | ΔPass@4 | ΔPass@8 | ΔPass@16 | Δsolved@16 |
|---|---:|---:|---:|---:|---:|
| 0.25× | +0.72 pp | +1.45 pp | +2.17 pp | +3.33 pp | +3.00 |
| 0.5× | **+1.06 pp** | **+2.41 pp** | **+3.61 pp** | **+5.19 pp** | **+4.67** |
| 1× | +0.53 pp | +1.20 pp | +2.05 pp | +3.70 pp | +3.33 |
| 2× | +0.76 pp | +1.98 pp | +3.26 pp | +4.81 pp | +4.33 |

0.25× 已经能稳定改善高 k 覆盖；2× 没有出现能力崩坏。

## Task-cluster bootstrap

### 0.25×

- Pass@4：+1.45 pp，95% CI [+0.27,+2.73]
- Pass@8：+2.17 pp，95% CI [+0.39,+4.24]
- Pass@16：+3.33 pp，95% CI [-0.37,+7.41]

### 0.5×

- Pass@1：+1.06 pp，95% CI [+0.25,+1.88]
- Pass@4：+2.41 pp，95% CI [+1.08,+3.87]
- Pass@8：+3.61 pp，95% CI [+1.47,+6.08]
- Pass@16：+5.19 pp，95% CI [+1.48,+9.63]

### 1×

- Pass@4：+1.20 pp，95% CI [+0.17,+2.31]
- Pass@8：+2.05 pp，95% CI [+0.54,+3.75]
- Pass@16：+3.70 pp，95% CI [+1.11,+7.04]

### 2×

- Pass@4：+1.98 pp，95% CI [+0.59,+3.51]
- Pass@8：+3.26 pp，95% CI [+1.07,+5.71]
- Pass@16：+4.81 pp，95% CI [+1.11,+9.26]

## 同随机方向、跨尺度配对比较

为排除不同随机方向造成的假剂量关系，对 seed 101/202/303 在相同 task 上直接比较尺度。

### 0.25× → 0.5×

- Pass@8：+1.44 pp，95% CI [-0.05,+2.99]
- Pass@16：+1.85 pp，95% CI [-0.74,+4.44]

方向上偏正，但区间仍跨 0。

### 0.5× → 1×

- Pass@8：**-1.56 pp，95% CI [-2.87,-0.47]**
- Pass@16：-1.48 pp，95% CI [-3.70,+0.37]

1× 在 Pass@8 上显著弱于 0.5×。

### 1× → 2×

- Pass@8：**+1.21 pp，95% CI [+0.13,+2.44]**
- Pass@16：+1.11 pp，95% CI [-0.74,+3.33]

2× 又恢复到更强的高 k 覆盖。

### 0.5× vs 2×

- Pass@8：-0.35 pp，95% CI [-1.66,+0.90]
- Pass@16：-0.37 pp，95% CI [-2.59,+1.85]

0.5× 与 2× 没有可靠差异。

## 行为漂移与剂量关系

一个非常关键的现象是：fixed-seed exact completion match 在所有非零 scale 都已经接近 68%–70%。

- 0.25×：平均 exact match ≈ 68.54%
- 0.5×：≈ 69.07%
- 1×：≈ 69.84%
- 2×：≈ 69.63%

因此行为漂移并没有随参数距离单调增加。

这与 Pass@k 的非单调关系一致：

> 很小的参数扰动就足以使随机采样轨迹大规模分叉，之后继续增大参数扰动并不会线性增加输出变化。

## Success-mass concentration

所有非零 scale 的平均 success HHI 都低于 SFT，effective task count 均高于 SFT。

0.5× 的平均 HHI 降幅最大之一，effective task count 平均 +1.82；2× 平均 +1.48。

## EXP-004E 最终结论

当前证据不支持简单的线性剂量—响应。

更符合数据的是：

1. 0.25× 已足以进入局部敏感区域；
2. 0.5×、1×、2× 都能恢复高 k 覆盖；
3. 0.5× 在当前三 seed 上点估计最强；
4. 1× 在同方向比较中反而弱于 0.5×，2× 又回升；
5. 0.5× 与 2× 无可靠差异；
6. 行为漂移从 0.25× 开始就近似饱和。

因此：

> SFT adapter 更像位于一个对随机采样极其敏感的局部区域。覆盖恢复不是参数距离的简单单调函数，而是在很小扰动后迅速触发并进入宽平台。

## 下一研究问题

现在不能继续把这种现象简单叫“参数空间不稳定”，因为当前行为指标来自 temperature=0.8 的随机采样。

下一步应区分：

- 参数扰动是否真的显著改变模型的条件分布 / logits；
- 还是极小 logit 改变被随机采样放大成大规模 autoregressive trajectory divergence。

下一实验优先做：

1. greedy decoding 的输出稳定性；
2. 固定 prompt 上 token-level KL / top-logit margin 变化；
3. 不同 temperature 下 fixed-seed behavior drift；
4. 将“模型分布变化”与“采样放大效应”分开。

这比继续增加 4× / 8× perturbation scale 更有研究价值。
