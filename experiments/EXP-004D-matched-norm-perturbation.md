# EXP-004D：Matched-Norm 随机参数扰动控制

## 状态

- 分支：exp/matched-norm-perturbation
- 扰动生成器：已实现
- 单元测试：已实现
- seed=101 smoke：待运行
- 3-seed 正式 n=16：待运行

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
