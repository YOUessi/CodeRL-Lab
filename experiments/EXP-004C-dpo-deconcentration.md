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


## No-op 因果控制结果

### 1. 普通 No-op：chosen == rejected，LoRA dropout=0.05

训练：

- 56 pairs；
- 21 optimizer steps；
- beta=0.1；
- learning rate=5e-7；
- chosen/rejected 文本完全相同；
- reward margin 始终为 0；
- train loss 约 0.6931。

但 adapter 相对 SFT：

- changed tensors：392 / 392；
- changed values：99.52%；
- relative L2：1.87e-4；
- max abs delta：4.84e-6。

固定 90题 ×16、同 seed 行为：

- completion exact match vs SFT：1001 / 1440 = 69.51%；
- changed rows：439；
- changed tasks：85 / 90。

多样性相对 SFT：

- unique fraction：+0.42 个百分点；
- empirical entropy：+0.0084 nats；
- modal fraction：-0.14 个百分点。

Pass@k 相对 SFT：

- Pass@1：+1.60 pp，95% CI [+0.63,+2.57]；
- Pass@4：+2.08 pp，95% CI [+0.69,+3.61]；
- Pass@8：+2.46 pp，95% CI [+0.56,+4.78]；
- Pass@16：+3.33 pp，95% CI [0,+7.78]。

因此极小参数漂移已经足以产生可检测的行为与覆盖变化。

### 2. No-op + LoRA dropout=0

关闭 LoRA dropout 后，adapter 漂移仍存在：

- changed tensors：392 / 392；
- changed values：99.52%；
- relative L2：1.82e-4；
- max abs delta：4.85e-6。

固定 n=16：

- completion exact match vs SFT：1009 / 1440 = 70.07%；
- changed rows：431；
- changed tasks：86 / 90。

多样性相对 SFT略下降：

- unique fraction：-0.35 个百分点；
- empirical entropy：-0.0043 nats；
- modal fraction：+0.28 个百分点。

Pass@k 相对 SFT：

- Pass@1：+0.07 pp，CI 跨0；
- Pass@4：+0.61 pp，CI 跨0；
- Pass@8：+1.40 pp，95% CI [-0.15,+3.36]；
- Pass@16：+2.22 pp，95% CI [0,+5.56]。

因此 dropout 会影响残余更新的方向/行为效果，但不是参数漂移存在的唯一来源。

### 3. Zero-learning-rate No-op

控制：

- chosen == rejected；
- adapter dropout=0；
- learning rate=0；
- 其它 DPO 流水线不变。

结果：

- grad norm 仍为非零（21/21 steps）；
- 但 adapter SHA 与原 SFT 完全相同；
- changed tensors：0；
- changed values：0；
- relative L2：0；
- exact tensor equality：true。

固定 n=16 重新生成：

- completion：1440 / 1440 与 SFT 完全一致；
- raw completion：1440 / 1440 完全一致；
- diversity 各指标完全一致；
- Pass@1/4/8/16 完全一致。

这同时排除了：

1. 保存/加载流程修改 adapter；
2. 固定 seed 的 GPU 生成本身存在运行间非确定性。

当前机制结论：

> chosen==rejected 在当前 TRL/DPO 数值实现中仍产生微小残余梯度；当 learning rate>0 时，这些残余梯度会积累成约 1e-4 相对 L2 的参数漂移，而这种微小漂移足以显著改变固定种子的采样路径。

因此 EXP-004B 中 Random-label / Semantic DPO 的覆盖恢复，必须与“非语义的小参数扰动效应”区分开，不能全部归因于偏好语义。

## Reverse-100 数据审计

56 / 56 prompt 完全相同；56 / 56 chosen/rejected 精确互换；候选文本集合逐对完全一致。

因此 Reverse-100 唯一变化是偏好方向。


# EXP-004C 正式结果

## 统一 n=16 结果

| Arm | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 | hidden mean |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 37.29% | 54.26% | 59.68% | 63.33% | 57 | 39.34% |
| Semantic DPO | 37.78% | 56.24% | **63.50%** | **68.89%** | **62** | 39.79% |
| Random-50 DPO | **38.47%** | 55.96% | 62.23% | 66.67% | 60 | **40.59%** |
| No-op, dropout=.05 | **38.89%** | **56.33%** | 62.14% | 66.67% | 60 | **40.90%** |
| No-op, dropout=0 | 37.36% | 54.87% | 61.08% | 65.56% | 59 | 39.41% |
| Zero-LR No-op | 37.29% | 54.26% | 59.68% | 63.33% | 57 | 39.34% |
| Reverse-100 DPO | 37.29% | 55.27% | 61.77% | **67.78%** | 61 | 39.34% |

## 固定种子行为一致性

相对 SFT 的 1440 个 completion：

| Arm | exact match | changed rows | changed tasks |
|---|---:|---:|---:|
| Zero-LR | **100.00%** | 0 | 0 |
| No-op .05 | 69.51% | 439 | 85 |
| No-op 0 | 70.07% | 431 | 86 |
| Random-50 | 69.44% | 440 | 87 |
| Semantic | 68.54% | 453 | 87 |
| Reverse-100 | 69.79% | 435 | 85 |

Zero-LR adapter 与原 SFT：

- SHA-256 完全相同；
- 392/392 tensor exact equality；
- 1440/1440 completion 完全相同；
- Pass@k 完全相同。

因此固定 seed 的生成链路本身是确定性的。约 30% 的输出变化来自真实参数扰动。

## No-op 残余梯度

chosen == rejected 的三条控制中，DPO 日志仍记录到非零 grad norm：

- No-op .05：mean grad norm ≈ 0.01344；
- No-op 0：≈ 0.01359；
- Zero-LR：≈ 0.01300。

三者 reward margin 都为 0。

区别只在参数更新：

- Zero-LR：lr=0，adapter 完全不变；
- No-op .05 / 0：lr>0，残余梯度积累成约 1.8e-4 相对 L2 的漂移。

因此当前 TRL/DPO 数值实现对 chosen==rejected 并非数值上严格零梯度。

## 参数更新方向

相对 SFT adapter 的 delta L2：

- Semantic：0.01451；
- Reverse-100：0.01445；
- Random-50：0.00759；
- No-op .05：0.00620；
- No-op 0：0.00603。

关键余弦：

- Semantic vs Reverse-100：**-0.9964**；
- Semantic vs No-op .05：0.0068；
- Semantic vs No-op 0：-0.0013；
- Random-50 vs No-op .05：0.0178；
- No-op .05 vs No-op 0：0.0354。

所以 Semantic 与 Reverse 几乎严格沿相反方向更新；No-op 残余更新则与两者近乎正交。

## Reverse-100 的关键现象

Reverse-100 相对 SFT：

- Pass@1：完全不变；
- hidden mean：完全不变；
- hidden-correct candidate 总数：**537 → 537，完全不变**；
- zero-correct tasks：33 → **29**；
- solved@16：57 → **61**；
- Pass@16：63.33% → **67.78%**。

Pass@16 的 paired bootstrap：

- delta：+4.44 pp；
- 95% CI：**[+1.11,+8.89] pp**；
- P(delta>0)=0.9824。

这意味着 Reverse 并没有创造更多正确候选，而是把同样数量的成功质量重新分配到了更多任务。

## 成功质量集中度

以每题 hidden-correct 候选数作为 success mass：

| Arm | total correct | zero tasks | HHI | effective task count |
|---|---:|---:|---:|---:|
| SFT | 537 | 33 | 0.02297 | 43.54 |
| Semantic | 544 | 28 | 0.02238 | 44.69 |
| Random-50 | 554 | 30 | 0.02227 | 44.90 |
| No-op .05 | 560 | 30 | **0.02203** | **45.38** |
| No-op 0 | 538 | 31 | 0.02273 | 43.99 |
| Zero-LR | 537 | 33 | 0.02297 | 43.54 |
| Reverse-100 | 537 | **29** | 0.02261 | 44.22 |

因此当前最合适的“去集中”定义不是 exact-output entropy，而是：

> 正确候选概率质量在任务之间的集中度下降，使相同或相近的总成功质量覆盖更多任务。

## exact-output diversity 并非统一上升

相对 SFT 的 empirical completion entropy delta：

- Semantic：+0.0111 nats；
- No-op .05：+0.0084；
- Reverse-100：+0.0042；
- No-op 0：-0.0043；
- Random-50：-0.0084；
- Zero-LR：0。

Random-50 即使 exact-output entropy 略下降，Pass@8/16 仍改善。因此覆盖恢复不能简单解释成“输出字符串更加多样”。

## EXP-004C 结论

当前证据排除了几个简单解释：

1. 不是生成运行间随机性：Zero-LR 1440/1440 完全复现；
2. 不是保存/加载扰动：Zero-LR adapter 与 SFT 逐值相同；
3. 不要求正确 preference direction：Reverse-100 也提高 Pass@16；
4. 不是简单的全局输出熵增加：Random-50 entropy 下降但高 k 覆盖改善；
5. No-op 残余更新与 Semantic/Reverse 方向近乎正交，也能改变覆盖。

最符合当前数据的机制是：

> SFT policy 位于一个对局部参数扰动非常敏感的概率分布区域。很小的 DPO/数值更新会重排若干低概率正确轨迹在不同任务上的概率质量，从而改变多样本覆盖；正确偏好语义会决定具体移动方向，但“覆盖恢复”本身并不要求正确语义方向。

## 下一因果问题

还剩最后一个关键替代解释：

> 是否任何同量级的随机参数扰动都能产生类似覆盖变化？

下一步进入 matched-norm random perturbation control：

- 从 SFT adapter 出发；
- 按 No-op .05 的每 tensor delta norm 匹配随机噪声幅度；
- 随机化方向；
- 多个随机 seed；
- 不经过 DPO 训练；
- 同一 n=16 固定生成与 Pass@k 评测。

如果随机噪声也稳定恢复高 k 覆盖，则“DPO 去集中”应改写为局部参数敏感性现象；如果随机噪声不稳定，而 DPO/No-op 残余更新稳定，则说明更新结构仍然重要。
