# EXP-004D：Matched-Norm 随机参数扰动控制

## 状态

- 分支：exp/matched-norm-perturbation
- 扰动生成器：已实现
- 单元测试：已实现
- seed=101 smoke：已通过
- 3-seed 正式 n=16：已完成

## 研究问题

EXP-004C 已经证明：

1. Zero-LR No-op 与 SFT adapter 完全一致，固定 seed 下 1440/1440 输出完全复现；
2. 只要 learning rate > 0，chosen==rejected 的 No-op DPO 仍产生约 1.8e-4 relative L2 的微小 adapter 漂移；
3. 这种微小漂移就足以让约 30% 固定 seed completion 改变；
4. Semantic / Random-50 / No-op / Reverse-100 都能在不同程度上恢复高 k 覆盖；
5. 正确 preference direction 并不是覆盖恢复的必要条件。

因此还剩最后一个关键替代解释：

> 是否任何同量级、任意方向的局部参数扰动都足以产生类似覆盖恢复？

## 控制设计

基准 adapter：

EXP-006A 1.7B SFT
SHA-256:
e1ea9a007a3e4213cdeb1ca6b1e12d29adb83594991421757d0d40b9168d0861

参考扰动：

EXP-004C No-op dropout=0.05 adapter - SFT adapter

其 global delta L2 约为：
0.006200807170745428

## Matched-Norm 规则

对每一个 LoRA tensor：

1. 计算 No-op05 相对 SFT 的该 tensor delta；
2. 记录该 tensor 的 delta L2 norm；
3. 从标准高斯随机方向采样 noise；
4. 将 noise 归一化；
5. 乘以该 tensor 的 No-op delta norm；
6. 加到 SFT tensor 上。

因此：

- 每 tensor 扰动幅度与 No-op05 匹配；
- 方向随机；
- 不经过 DPO；
- 不使用 preference pair；
- 不使用 optimizer；
- 不涉及 reward；
- 只有 seed 决定随机方向。

正式 seeds：

- 101
- 202
- 303

## 预注册判断

### 支持“局部参数敏感性”

若多个随机 seed 都出现：

- Pass@8 / Pass@16 相对 SFT 正向；
- solved@16 增加；
- success-mass HHI 下降 / effective task count 上升；

且量级与 No-op / Random-50 / Reverse-100 相近，则说明：

> 高 k 覆盖恢复不需要 DPO 特有更新结构，普通同量级局部扰动即可触发。

### 支持“DPO 更新结构重要”

若：

- 随机扰动 seed 间高度不稳定；
- 平均没有稳定恢复高 k 覆盖；
- 或多数 seed 明显破坏覆盖；

而 DPO/No-op/Reverse 更稳定，则说明：

> 不只是参数扰动幅度，DPO 更新的结构、梯度几何或低秩方向仍然重要。

## 评测

每个 seed：

- MBPP validation 90 题；
- n=16；
- 与 EXP-004C 同 seed=42 生成；
- Pass@1/4/8/16；
- solved@16；
- hidden mean；
- exact completion diversity；
- success-mass HHI；
- 相对 SFT paired bootstrap。

另外记录：

- output adapter SHA-256；
- global delta L2；
- per-tensor norm ratio；
- 与 No-op05 delta 的 cosine；
- 三个随机 seed 彼此的 cosine。

## 记录要求

所有操作同时写入：

- experiments/EXP-004D-matched-norm-perturbation.md
- results/exp004d/
- docs/daily/2026-10-07.md

失败实验、并发问题、重跑和修复都保留。


# 正式 3-seed n=16 结果

三条随机扰动都满足：

- per-tensor 扰动 L2 与 No-op05 对应 tensor delta 匹配；
- global delta L2 ≈ 0.006200812；
- 与 No-op05 delta cosine ≈ 0；
- 三个随机 seed 彼此 cosine ≈ 0。

因此是“同幅度、随机方向”的纯参数扰动控制。

## 单 seed 结果

| Seed | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 | ΔPass@16 vs SFT |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 37.43% | 54.83% | 61.30% | 66.67% | 60 | +3.33 pp |
| 202 | 38.47% | 55.89% | 62.10% | 67.78% | 61 | +4.44 pp |
| 303 | 37.57% | 55.66% | 61.78% | 66.67% | 60 | +3.33 pp |
| SFT | 37.29% | 54.26% | 59.68% | 63.33% | 57 | — |

三个随机 seed 的 Pass@4 / Pass@8 / Pass@16 增益全部为正。

### seed 202 paired bootstrap

- Pass@4：+1.63 pp，95% CI [+0.24,+3.11]
- Pass@8：+2.43 pp，95% CI [+0.43,+4.81]
- Pass@16：+4.44 pp，95% CI [+1.11,+8.89]

seed101 / 303 的高 k 点估计同样为正，但单 seed 区间更宽。

## 三 seed 任务级聚合 bootstrap

先对每个 task 的三个 seed 增益取平均，再对 90 个 task 做 20,000 次 paired bootstrap：

| 指标 | mean delta | 95% CI | P(delta>0) |
|---|---:|---:|---:|
| Pass@1 | +0.53 pp | [-0.23,+1.32] | 0.9154 |
| Pass@4 | **+1.20 pp** | **[+0.17,+2.31]** | 0.9896 |
| Pass@8 | **+2.05 pp** | **[+0.54,+3.75]** | 0.9981 |
| Pass@16 | **+3.70 pp** | **[+1.11,+7.04]** | 0.9978 |

因此平均高 k 覆盖改善具有明确正区间，而 Pass@1 并没有同量级提升。

## Success-mass concentration

SFT：

- HHI = 0.0229671
- effective task count = 43.54

随机扰动：

- seed101 HHI = 0.0227832，effective tasks = 43.89
- seed202 HHI = 0.0221559，effective tasks = 45.13
- seed303 HHI = 0.0223759，effective tasks = 44.69

三个 seed 均：

- HHI 下降；
- effective task count 上升；
- solved@16 增加 3–4 题。

## 行为漂移

相对 SFT 固定 seed 的 1440 个 completion：

- seed101 exact match = 69.79%，435 条改变；
- seed202 exact match = 68.89%，448 条改变；
- seed303 exact match = 70.83%，420 条改变。

三条随机扰动虽然参数方向几乎互相正交，但都能触发约 29%–31% 的固定 seed 输出变化。

## EXP-004D 机制结论

预注册“局部参数敏感性”条件得到满足：

1. 三个随机方向的 matched-norm 扰动全部提高 Pass@8 / Pass@16；
2. solved@16 全部增加；
3. success-mass HHI 全部下降；
4. effective task count 全部增加；
5. 与 No-op05 参数更新方向 cosine ≈ 0；
6. 完全不经过 DPO、preference pair、optimizer 或 reward。

因此当前最符合证据的结论是：

> SFT 后的高 k 覆盖恢复并不需要 DPO 特有更新结构。Qwen3-1.7B SFT adapter 位于一个对极小局部参数扰动高度敏感的区域；任意同量级随机方向都可能重新分配跨任务成功概率质量，缓解 SFT 的概率集中。

这进一步解释了 EXP-004C 中 Semantic / Random / No-op / Reverse 都能出现覆盖恢复。

但需要严格限制结论：

- 当前只测试 3 个预注册随机 seed；
- 结论说明“随机同量级扰动足以触发效应”，不等于所有随机扰动都必然提升；
- seed-level 样本数仍小，不能把 task-level bootstrap 当成无限随机方向上的总体统计。

## 工程问题与修复

Tang 开始统一分析时，本地 exp/matched-norm-perturbation 落后远端 5 个提交，导致 scripts/run_exp004d_analysis.sh 不存在。

处理：

1. 确认已有三 seed artifacts 已完整保存在 Tang；
2. git fetch origin；
3. reset 到 origin/exp/matched-norm-perturbation 最新提交；
4. 保留 artifacts 不动；
5. 112 个单元测试全部通过；
6. 重新运行统一分析脚本。

这次问题不影响模型生成或评测结果，只影响后处理脚本可见性。

## 下一步

EXP-004D 已回答“DPO 更新结构是否必要”：当前证据倾向否。

下一问题应该转向：

> 为什么 SFT adapter 处在这种局部高敏感区域？覆盖改善是否存在扰动幅度的剂量关系？

比继续比较 preference 标签更有价值的后续控制：

- 0.25× / 0.5× / 1× / 2× matched perturbation scale；
- 固定多 seed；
- 检查 Pass@k、success HHI、行为漂移随扰动幅度的响应曲线。
