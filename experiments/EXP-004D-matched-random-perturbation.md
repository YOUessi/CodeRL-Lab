# EXP-004D：同范数随机参数扰动控制

## 研究问题

EXP-004C 已证明：

- Zero-LR 完全复现 SFT；
- No-op / Random-50 / Reverse-100 等非零小更新都会改变高 k 覆盖；
- 正确 preference direction 不是覆盖恢复的必要条件；
- exact-output entropy 也不是统一解释。

最后一个关键替代解释是：

> 是否任意同量级的局部参数扰动，都能在 SFT 附近产生类似覆盖恢复？

## 控制设计

基线：Qwen3-1.7B SFT adapter。

参考扰动：EXP-004C No-op(dropout=0.05) adapter 相对 SFT 的 delta。

随机控制：

- 对每个 LoRA tensor 单独计算 No-op delta 的 L2 norm；
- 随机生成同 shape 高斯方向；
- 将随机方向归一化到与该 tensor No-op delta 完全相同的 L2 norm；
- 不经过 DPO 训练；
- 不使用偏好数据；
- 保留原 SFT adapter config；
- 使用 3 个独立随机 seed：101 / 202 / 303。

## 预注册检查

每个随机 seed 必须满足：

1. tensor key / shape 与 SFT 完全一致；
2. 每 tensor delta L2 与 No-op 对应 tensor 匹配；
3. global delta L2 与 No-op 匹配；
4. 与 No-op delta 的 global cosine 接近 0，而不是偶然复制其方向；
5. 固定 n=16 评测使用与 EXP-004C 完全相同的 90 个任务、temperature、top-p、seed。

## 正式评测

每个随机 seed：

- 90 tasks × 16 samples = 1440 completions；
- Pass@1/4/8/16；
- solved@16；
- hidden mean；
- completion diversity；
- 与 SFT 的 exact-match fraction；
- 与 SFT 的 paired bootstrap。

## 判定

若 3 个 matched-norm 随机方向都稳定恢复高 k 覆盖：

> EXP-004C 的“DPO 去集中”应主要解释为 SFT 附近的局部参数敏感性。

若随机方向效果高度不稳定，而 No-op / Random / Reverse 等结构化更新稳定有效：

> 更新方向结构仍然重要，不能简化为任意小扰动。

若随机扰动大多破坏性能：

> DPO/No-op 残余更新虽然语义方向不关键，但仍可能具有特殊的优化器/梯度结构。
