# EXP-003：可验证奖励 GRPO 基线

## 状态

**环境与接口核验：完成。**  
**2-step GRPO 冒烟：完成。**  
**374 题完整 1 epoch GRPO：完成。**  
**90 题 validation：完成。**

## 研究目的

在 EXP-002 的监督微调模型基础上加入**可验证奖励强化学习**，建立：

```text
Base
vs
SFT
vs
SFT + GRPO
```

的第一条完整后训练对照。

第一阶段不追求新算法，只要求：

1. 奖励严格只使用 train split 公共测试；
2. 隐藏测试完全不进入训练；
3. GRPO 训练过程可复现、可记录；
4. 与 EXP-002 使用完全相同的 validation Pass@k 流水线；
5. 记录探索、多样性、奖励方差和有效样本比例。

## 初始化策略

基础模型：

`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`

SFT adapter：

`7fcb2b0ca7608be980fb4cbae5158241886ac67f3cc2f5b855cf34531c5982f7`

GRPO 启动前强制校验该 SHA-256，不一致则拒绝训练。

## 数据

固定 MBPP-v1：

`google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c`

训练时只使用：

- train split prompt；
- train split 公共测试。

训练数据结构中**不包含 hidden_tests**。

## 第一版奖励

```text
R = 0.10 × 语法正确
  + 0.80 × 公共测试通过率
  + 0.10 × 公共测试全部通过
```

## GRPO 配置

- `loss_type = "grpo"`，显式覆盖 TRL 1.14.1 默认的 `dapo`；
- `beta = 0`；
- `num_generations = 4`；
- `scale_rewards = "group"`；
- temperature = 0.8；
- top-p = 0.95；
- max completion length = 256；
- per-device batch = 4；
- gradient accumulation = 2；
- generation batch = 8；
- 每个优化步包含 2 个独立 prompt 组；
- learning rate = 1e-6；
- 1 epoch；
- warmup = 10 optimizer steps；
- seed = 42。

## 工程问题与修复

### 1. GRPOTrainer 缺少 requests

TRL 1.14.1 导入 GRPO trainer 时会加载 vLLM client，环境缺少 `requests`。

修复：固定 `requests==2.34.2`。

### 2. YAML 的 no 被解析成 False

```yaml
save_strategy: no
```

被 PyYAML 解释成布尔值。改为：

```yaml
save_strategy: "no"
```

### 3. Docker timeout 留下孤儿容器

原执行器 timeout 只结束 `docker run` CLI，候选死循环容器可能继续运行。

修复：

- 每次容器设置唯一名称；
- 加 `coderl_lab=1` 标签；
- 超时后强制 `docker rm -f`；
- 真实死循环测试验证：
  `timed_out=True`，残留容器数 = 0。

历史遗留 9 个容器已清理。

### 4. 预热步数计算

374 个唯一任务、4 generation、batch 4、gradient accumulation 2：

每个 optimizer update 实际覆盖 2 个唯一 prompt，因此：

```text
ceil(374 / 2) = 187 optimizer steps
```

5% warmup：

```text
ceil(187 × 0.05) = 10
```

### 5. 奖励并行

completion reward 从串行执行改为 4 worker 并行。

固定 2-step 冒烟中：

- reward / reward_std / entropy / grad_norm 完全一致；
- runtime 从 12.18 s 降到 7.39 s。

说明只是吞吐优化，不改变算法结果。

### 6. 持久化逐步训练历史

首次正式运行到 21/187 step 时主动中止，因为发现没有保存逐步：

- reward；
- reward std；
- frac_reward_zero_std；
- entropy；
- grad norm；
- completion length；
- step time。

随后增加 `log_history.json` 后，从固定代码重新完整训练。

该 21-step 运行不算正式结果。

## 正式训练结果

最终 GRPO adapter SHA-256：

`1140fab138e31882404933ce4ba91f40a483e0ec9db18b4acc999383fb34529b`

| 指标 | 数值 |
|---|---:|
| unique train tasks | 374 |
| optimizer steps | 187 |
| num generations | 4 |
| train runtime | 751.10 s |
| steps/s | 0.249 |
| 峰值 GPU allocated | 2,704,101,888 bytes |
| 峰值 GPU reserved | 6,958,350,336 bytes |
| 最终运行容器残留 | 0 |

GRPO 的 policy-gradient loss 接近 0 是组相对目标的正常数值表现，不能像 SFT 交叉熵一样直接解释训练好坏。

## 训练动力学

187 个训练 step 的均值：

| 指标 | 均值 |
|---|---:|
| reward | 0.4342 |
| reward std | 0.3314 |
| **frac reward zero std** | **0.5214** |
| entropy | 0.2770 |
| grad norm | 0.9258 |
| completion mean length | 55.83 |
| step time | 4.01 s |
| public test reward | 0.3753 |
| syntax reward | 0.9973 |

### 前四分之一 vs 后四分之一

| 指标 | 前 25% | 后 25% | 变化 |
|---|---:|---:|---:|
| reward | 0.4315 | 0.5236 | +0.0921 |
| public test reward | 0.3723 | 0.4742 | +0.1019 |
| all-public-pass reward | 0.3370 | 0.4457 | +0.1087 |
| entropy | 0.2803 | 0.2703 | -0.0100 |
| frac reward zero std | 0.5109 | 0.5109 | **0** |

训练后期公共测试成功率提高，但**无组内差异的比例没有下降**。

### 无效更新信号

- 187 步中 grad norm = 0：**52 步（27.81%）**；
- 整个 generation batch 的 reward std = 0：28 步（14.97%）；
- 平均 `frac_reward_zero_std = 52.14%`。

注意：

一个 batch 可以有非零全局 reward std，但其中每个 prompt 组内部奖励完全相同。GRPO 做的是**组内**归一化，因此这种 batch 仍可能没有任何策略梯度。

这解释了为什么：

> 单纯看 batch reward std 不足以判断 GRPO 是否有有效学习信号。

## 90 题 validation

完全沿用 EXP-002：

- 同 90 道 MBPP validation；
- 每题 4 候选；
- 同 prompt；
- 同 temperature/top-p；
- 同随机种子；
- 同代码归一化；
- 同 Docker 执行器；
- 同隐藏测试。

### 三模型主结果

| 指标 | Base | SFT | SFT + GRPO |
|---|---:|---:|---:|
| Pass@1 | 13.61% | 25.56% | **27.22%** |
| Pass@4 | 38.89% | **43.33%** | 42.22% |
| 语法失败 | 182 / 360 | 0 / 360 | 0 / 360 |
| 平均公共测试通过率 | 14.17% | 27.78% | **30.56%** |
| 平均隐藏测试通过率 | 14.86% | 28.19% | **29.86%** |
| 公共测试全通过候选 | 51 | 100 | **110** |
| 隐藏测试全通过候选 | 49 | 92 | **98** |
| 至少一个候选正确的题 | 35 / 90 | **39 / 90** | 38 / 90 |
| 4/4 候选全部正确的题 | 0 | 8 | **12** |

### GRPO 相对 SFT

- Pass@1：**+1.67 个百分点**
- Pass@4：**-1.11 个百分点**
- 平均公共测试通过率：+2.78 个百分点
- 平均隐藏测试通过率：+1.67 个百分点
- 语法失败：保持 0

按题：

- 10 题正确候选数增加；
- 4 题正确候选数减少；
- 76 题不变；
- 2 题从 SFT 的 4 次全错变成 GRPO 至少成功 1 次；
- 3 题从 SFT 有成功候选变成 GRPO 4 次全错。

## 关键结论

### 1. 第一版 GRPO 有真实但很小的 Pass@1 增益

SFT → GRPO：

```text
25.56% → 27.22%
```

说明可验证奖励强化学习在这个 0.6B 设置下仍能进一步提升单次成功概率。

### 2. Pass@4 没有同步提高

```text
43.33% → 42.22%
```

这说明 GRPO 的收益不能简单解释成“能力集合扩大”。

至少在当前 k=4 行为上，更像是：

> 对已有输出分布做进一步概率重排。

真正的能力边界结论仍需要更大的 k 和外部基准。

### 3. GRPO 产生了更强的确定性成功

4/4 全正确的题：

```text
SFT: 8
GRPO: 12
```

同时至少有一个候选正确的题：

```text
SFT: 39
GRPO: 38
```

这和 Pass@1 上升、Pass@4 下降是相互一致的：

> 部分已经会的题变得更稳定，但整体覆盖的题并没有扩大。

### 4. 最大训练效率问题已经被直接观测到

平均约一半 prompt group：

```text
frac_reward_zero_std ≈ 52.14%
```

没有组内奖励差异。

这意味着大量 rollout 对纯 GRPO 没有相对优势信号。

因此下一阶段最自然的研究问题不是盲目增加训练轮数，而是：

> **能否优先训练“当前模型有时会、有时不会”的能力边界样本？**

这直接对应动态采样（Dynamic Sampling）/课程学习方向。

## EXP-003 结论边界

现在可以说：

- 在相同 0.6B 模型、固定 MBPP-v1 和统一评测下，SFT 后继续做纯 GRPO 可小幅提高 Pass@1；
- 该增益没有同步提高 Pass@4；
- GRPO 训练中存在大量零组内方差 prompt；
- 动态采样有直接实验动机。

现在不能说：

- GRPO 扩展了基础模型的新能力边界；
- GRPO 一定优于 SFT；
- 当前 MBPP 结果可以代表无污染泛化；
- 0.6B 行为可直接外推到 1.7B / 4B。

## 下一步

优先进入**能力边界动态采样实验**：

1. 在当前 SFT policy 上预采样每个训练题的成功率；
2. 区分全错、混合、全对题；
3. 对比随机训练与只保留有组内奖励差异的动态训练；
4. 在相同 rollout budget 下比较：
   - 有效梯度比例；
   - wall-clock；
   - Pass@1；
   - Pass@4；
   - 覆盖题数；
   - entropy；
   - reward zero-std 比例。

过程级奖励实验保留为后续独立变量，不与动态采样同时修改。
