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
