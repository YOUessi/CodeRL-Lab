# EXP-004K：Held-Out Verifier-Gated Bottleneck Replication

## 状态

- 分支：exp/heldout-verifier-gated-bottleneck
- 数据：MBPP test 500 tasks
- 规则：完全冻结自 EXP-004J
- held-out runner：已实现
- smoke：已通过
- 正式500题：已完成，held-out primary 通过

## 研究目的

EXP-004J 在反复用于机制开发的 MBPP validation 90题上观察到：

- SFT baseline hidden accuracy：40.00%
- public-gated low-destabilize：43.33%
- +3.33 pp
- wrong→correct=3
- correct→wrong=0
- high-margin control=0 change

但 primary 95% bootstrap CI=[0,+7.78]，没有严格满足预注册的 CI lower bound >0。

为了避免继续在同一 validation 上调参，本实验冻结全部规则，直接在此前没有参与 EXP-004F-J 方法设计的 MBPP test 500题上做 held-out replication。

## 冻结规则

不再修改：

- model：Qwen3-1.7B-Base + EXP-006A SFT adapter
- greedy decoding
- max_new_tokens=512
- primary window=128
- low threshold=0.05
- high threshold=0.20
- bias=-0.25
- gate：SFT baseline public tests 未全部通过
- gated-low / gated-high / always-low 定义
- task bootstrap iterations=20,000
- bootstrap seed=42

## 数据

固定：

- data/generated/mbpp-v1/test_tasks.jsonl
- 500 tasks
- task_id unique 500/500
- public tests 500/500 non-empty
- hidden tests 500/500 non-empty

test split 在本实验开始前没有用于 threshold / bias / gate 设计。

## 无 hidden 泄漏的数据流

### Phase 1：SFT baseline + margin trace

GPU 只生成：

- baseline raw completion
- reference token ids
- first128 margin profile
- low/high intervention positions

不运行 hidden tests。

### Phase 2：public-only gate

只对 baseline code 运行 public tests。

生成 gate：

public_fail AND low_position exists

该阶段的 artifact 不包含 hidden outcome。

### Phase 3：SFT second-pass intervention

保持同一 SFT adapter：

- gated_low_destabilize：gate=true 时 low bottleneck -0.25
- gated_high_destabilize：gate=true 时 nearest high-margin -0.25
- always_low_destabilize：所有 eligible task low bottleneck -0.25
- baseline：不改

### Phase 4：最终 outcome

所有生成完成以后，才统一运行 hidden tests。

hidden outcome 不参与 gate、position 或 bias 选择。

## Primary hypothesis

全500题：

gated_low hidden correctness - baseline hidden correctness

预注册成功标准：

- paired task bootstrap 95% CI lower bound >0

## Secondary controls

### Position specificity

gated_low - gated_high

### Gate necessity

gated_low - always_low

同时报告：

- wrong→correct
- correct→wrong
- public-pass task 是否被 gated policy 改写
- eligible / gate-triggered task 数

## 解释

如果 held-out primary CI >0：

说明 public-test-gated low-margin escape 在未参与方法开发的数据上得到复制。

如果点估计正但 CI 跨0：

只能认为方向一致但证据不足。

如果无效或负向：

EXP-004J validation 结果应视为开发集特定现象，不继续优化该 decoding rule。


## 12-task smoke

代码/数据流 smoke 已完成：

- CPU tests：140/140 通过；
- runner tasks：12；
- eligible：5；
- gate triggered：2；
- runner 明确记录 hidden_tests_accessed=false；
- public-pass task 中 gated-low / gated-high 均不改写；
- always-low 可改写 public-pass task，符合 gate necessity control；
- baseline hidden correct：8/12；
- gated-low：8/12；
- gated-high：8/12；
- always-low：7/12。

Smoke 只用于工程与无泄漏检查，不用于方法选择。

## 正式 500-task 启动

正式运行提交：

`ec073e974575f83e5234f68b5d8e87a4e7532654`

冻结规则与本文件预注册完全一致。启动前：

- 140/140 CPU tests passed；
- Tang GPU 正常；
- Docker 无遗留；
- PR #26 CPU CI passed；
- 正式运行中不修改 low/high threshold、bias、gate、window 或 bootstrap 配置。

正式 hidden outcome 只会在 runner/public-only gate/second-pass 全部冻结后统一读取。


# 正式 500-task held-out 结果

正式 runner 已完成全部 500 个 MBPP test task，并且最终 artifact 明确记录：

`hidden_tests_accessed = false`

hidden outcome 只在 baseline / public-only gate / second-pass intervention 全部冻结后统一评测。

## 全 500 题主要结果

| 条件 | Hidden correct | Accuracy | Δ vs baseline |
|---|---:|---:|---:|
| Baseline SFT greedy | 207 / 500 | 41.40% | — |
| **Gated low-margin destabilize** | **215 / 500** | **43.00%** | **+1.60 pp** |
| Gated high-margin control | 207 / 500 | 41.40% | 0 |
| Always-low, no gate | 207 / 500 | 41.40% | 0 |

Primary paired task bootstrap（20,000 次）：

- gated-low - baseline：**+1.60 pp**；
- 95% CI：**[+0.40,+3.00] pp**；
- P(delta>0)=0.98995。

因此 held-out primary hypothesis 正式通过预注册成功标准：95% CI lower bound > 0。

## 转移计数

### Gated low vs baseline

- wrong→correct：**10**；
- correct→wrong：**2**；
- correct→correct：205；
- wrong→wrong：283。

### High-margin control

- wrong→correct：0；
- correct→wrong：0；
- 完全无行为收益。

### Always-low without gate

- wrong→correct：10；
- correct→wrong：10；
- 净提升 0。

这说明 gate 不是可有可无的工程细节：同样 low-margin destabilize，如果对所有 eligible task 无条件使用，收益与伤害完全抵消。

## Gate-triggered 子集

实际触发 second pass：159 / 500。

在这 159 题中：

- baseline correct：7 / 159 = 4.40%；
- gated-low：15 / 159 = **9.43%**；
- 提升：**+5.03 pp**；
- 95% CI：**[+1.26,+9.43] pp**。

转移：

- wrong→correct：10；
- correct→wrong：2。

## Gate 与 hidden-wrong 的关系

仅 public-fail（不要求 bottleneck）在 500 held-out 上：

- public-fail tasks：280；
- baseline hidden-wrong：293；
- public-fail ∩ hidden-wrong：269；
- precision：**96.07%**；
- recall：**91.81%**。

真正的 actual gate = public-fail AND low bottleneck exists：

- selected：159；
- hidden-wrong：152；
- hidden-correct：7；
- precision：**95.60%**；
- recall：51.88%。

所以 gate 是一个高 precision、低 recall 的保守干预规则：只在非常可能错误、且确实存在 low-margin bottleneck 的任务上触发。

## 位置特异性

Gated-low 与 gated-high 的差异：

- +1.60 pp；
- 95% CI：**[+0.40,+3.00] pp**。

同样的 -0.25 logit 干预放在 high-margin control 位置完全没有收益，说明不是“任意修改 logits”即可。

## Gate necessity

Gated-low 与 always-low 的差异：

- +1.60 pp；
- 95% CI：**[+0.60,+2.80] pp**；
- P(delta>0)=0.99985。

public-pass 的 220 个任务：

- gated-low changed：0；
- gated-high changed：0；
- always-low changed：90。

因此 gate 同时承担两件事：

1. 用 public verifier 识别“值得逃逸”的错误轨迹；
2. 保护已经能通过 public tests 的轨迹不被无谓扰动。

# EXP-004K 结论

EXP-004H-I-J-K 的证据链现在闭环：

1. local low-margin bottleneck 可预测 perturbation susceptibility；
2. 对 low-margin token 的受控 stabilize/destabilize 可以因果改变 trajectory；
3. SFT reference 错误时，destabilize 可解锁 alternative correct trajectory；
4. public tests 可作为不依赖 hidden outcome 的高 precision gate；
5. SFT-only verifier-gated two-pass decoding 在 90-task validation 上方向为正但 CI 下界=0；
6. 冻结全部规则后，在独立 MBPP test 500-task 上复制成功：**+1.60 pp，95% CI [+0.40,+3.00]**。

因此当前可以正式说：

> **局部 low-margin bottleneck 是可干预的自回归决策闸门；当外部可验证信号表明当前 greedy 轨迹可能错误时，对该 bottleneck 做小幅定向 destabilize，可以在不重新训练模型的情况下解锁替代轨迹，并在 held-out 500 题上获得统计显著的正确率提升。**

同时，high-margin control 和 ungated always-low 都没有净收益，说明位置特异性与 verifier gate 都是必要组成部分。

## 下一步

不继续在 MBPP 上调 threshold / bias / window。

优先做外部分布验证：

1. LiveCodeBench / 时间更新代码任务；
2. 如果任务格式允许，复现 public-verifier-gated bottleneck escape；
3. 或迁移到仓库级软件工程任务，使用可执行测试作为 verifier；
4. 检查 low-margin bottleneck + verifier gate 是否跨数据分布、跨任务粒度成立。
