# EXP-004C：DPO 去集中机制

## 状态

- Semantic repair DPO：复用 EXP-004B
- Random-label 50% DPO：复用 EXP-004B
- Reverse-label 100% DPO：待运行
- No-op DPO：待运行
- n=16 统一评测：待运行

## 核心问题

EXP-004B 发现：

- 正确 semantic DPO 能恢复 SFT 多样本覆盖；
- 50% 随机标签 DPO 也能恢复覆盖；
- semantic DPO 相对 random control 没有显著 Pass@k 优势。

因此当前研究问题不是“修 import”，而是：

> 为什么一个很小的 DPO 更新会缓解 SFT 的概率集中？

## 四臂因果控制

固定相同 56 个 prompt / candidate pair 文本、相同 SFT 初始化、相同 21 optimizer steps、相同 beta / learning rate / seed。

1. Semantic：正确 minimal-import preference，已有。
2. Random-50：随机翻转 28/56 标签，已有。
3. Reverse-100：全部 chosen/rejected 反转。
4. No-op：chosen == rejected，理论 DPO gradient 为 0。

## 预注册假设

### H0：训练流水线扰动

如果 no-op DPO 的 adapter 权重与 SFT 不完全一致，或固定种子 n=16 输出发生系统变化，则必须先解释数值/训练流水线扰动，不能把其它控制解释为 DPO 机制。

期望：

- no-op adapter tensor exact equality = true；
- no-op completion 与 SFT 同 task/sample 完全一致；
- Pass@k 完全一致。

### H1：偏好方向是否必要

如果 Reverse-100 也像 Random-50 一样提高多样本覆盖，则覆盖恢复不依赖正确 preference direction。

如果 Reverse-100 明显破坏覆盖，而 Random-50 恢复，则说明“相互冲突的 pairwise update / margin regularization”可能是关键，而不是任意方向更新。

### H2：去集中是否真实发生

除 Pass@k 外，固定 n=16 统计：

- exact unique completions / task；
- empirical completion entropy；
- normalized entropy；
- modal completion fraction；
- duplicate fraction。

如果 random/reverse 的覆盖恢复伴随：

- entropy 上升；
- modal fraction 下降；
- unique completion count 上升；

则支持 deconcentration。

如果 Pass@k 变化但 exact-output diversity 不变，需要寻找更细的 token-level probability mechanism。

## 评测

统一 90 个 MBPP validation tasks × 16 samples：

- Pass@1/4/8/16；
- solved@16；
- hidden mean；
- completion diversity；
- paired bootstrap。

后续若机制信号明确，再决定是否增加 3 个 random-label seeds。
