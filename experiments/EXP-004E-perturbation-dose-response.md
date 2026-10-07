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
