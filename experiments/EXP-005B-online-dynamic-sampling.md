# EXP-005B：在线策略依赖的动态采样

## 状态

**设计：完成。**  
**代码：第一版完成。**  
**GPU 冒烟：已通过。**  
**正式实验：已完成。**

## 来自 EXP-005A 的问题

005A 已证明：

- 静态 boundary 选择显著降低 zero-grad；
- 但 validation 没有优于同大小随机子集；
- 初始 mixed 题在训练中会变成 flat。

所以边界不是任务的固定属性，而是：

```text
boundary(task, current_policy)
```

## 005B-v1：在线反馈 sampler

第一版不侵入 TRL 的 reward→advantage→loss tensor 内核。

每个 generation batch 当前包含 2 个独立 prompt group：

```text
slot A：exploration
  优先从从未在线观察过的任务中选

slot B：exploitation
  从“最近一次在线 reward spread > 0”的 mixed pool 中选
```

如果当前还没有 mixed task，则两个 slot 都探索。

每次 reward 计算完成后立即更新：

- task 当前状态：unknown / mixed / flat；
- last reward spread；
- last mean reward；
- mixed↔flat 转移；
- 选择次数；
- exploit / explore 次数。

如果一个 mixed task 后来重新采样变 flat，它立即退出 exploit pool。

## 为什么先做 v1，而不是直接 batch 内 drop+refill

TRL 1.14.1 的 GRPO 内部把：

```text
生成
→ token padding/mask
→ old/ref logp
→ rewards
→ group normalization
→ advantages
→ buffered gradient-accumulation slices
```

紧密绑定。

直接在 rewards 后删除/补 completion 会同时影响：

- prompt/completion tensor 对齐；
- completion mask；
- old logp；
- ref logp；
- buffered inputs；
- multi-process process_slice。

005B-v1 只替换 train sampler，不修改这些内核，因此结果更容易解释和复现。

若 v1 仍有大量 flat group，再单独实现 005B-v2 batch 内 drop+refill。

## 固定 raw rollout budget

与 EXP-003 保持：

- 187 optimizer steps；
- 每步 2 prompt groups；
- 每组 4 completions；
- 总 prompt groups = 374；
- 总 RL completions = 1496；
- 无离线 screening 开销。

因此 005B-v1 可以直接回答：

> 在相同原始生成预算下，在线反馈分配 prompt 是否能提高有效更新比例？

## 探索 / 利用

第一版：

- exploit_fraction = 0.5；
- generation batch 的 2 个 group 中，理想状态下 1 个 exploit + 1 个 explore；
- exploration 优先覆盖 unknown task；
- exploit 从当前 mixed pool 中选择被使用次数最少的任务；
- 同一个 generation batch 不重复 task。

这显式保留任务多样性，避免 005A 只围绕固定 181 个初始边界题训练。

## 主要对照

- EXP-003：全量均匀随机；
- EXP-005A random subset；
- EXP-005A static boundary；
- EXP-005B online adaptive。

训练指标：

- frac_reward_zero_std；
- zero-grad fraction；
- effective steps；
- unique selected tasks；
- exploit / explore 比例；
- mixed→flat / flat→mixed 转移；
- completions per effective step；
- entropy；
- runtime。

验证指标：

- Pass@1；
- Pass@4；
- hidden mean；
- solved tasks；
- 4/4 全正确任务。

## 成功判据

005B-v1 至少需要满足：

1. 相同 1496 RL completions 下，有效更新率优于 EXP-003；
2. validation 不明显退化；
3. 比 005A static boundary 保留更多在线任务多样性；
4. 在线状态确实产生 mixed→flat / flat→mixed 转移，而不是退化成静态子集。

如果训练效率提升但 validation 仍无增益，则说明下一步需要研究：
- 更合理的 exploit_fraction；
- group 内 drop+refill；
- 或 reward / process supervision，而不是继续单纯采样。


## GPU 冒烟结果

代码提交：

`32c208985668ee2b63517dd941fbf5ea90002af2`

设置：

- train tasks：32；
- max steps：4；
- num_generations：4；
- generation batch：2 prompt groups；
- exploit_fraction：0.5；
- 无离线 screening。

结果：

| 指标 | 数值 |
|---|---:|
| train runtime | 11.50 s |
| zero-grad steps | 1 / 4 |
| mean frac_reward_zero_std | 0.50 |
| peak allocated | 2,032,813,056 bytes |
| peak reserved | 3,118,465,024 bytes |
| actual rollout groups | 8 |
| actual rollout completions | 32 |
| total sampler selections | 10 |
| prefetched but unobserved groups | 2 |
| exploit selections | 3 |
| explore selections | 7 |
| unique selected tasks | 7 |
| unique observed tasks | 6 |

在线状态在 4 步中实际观察到：

- mixed groups：4；
- flat groups：4；
- observed mixed fraction：0.5。

状态转移：

```text
unknown -> mixed : 3
unknown -> flat  : 3
mixed   -> mixed : 1
mixed   -> flat  : 1
```

这验证了 005B 的核心假设：

> boundary 是 current-policy-dependent 状态，而不是静态 task 属性。

特别是 `mixed -> flat` 已在极小冒烟中出现，说明 exploit pool 会真实动态变化。

另外 sampler 会因 DataLoader/Trainer 预取产生尚未实际 rollout 的 selection。当前实现已分别记录：

- total group selections；
- groups observed；
- prefetched unobserved selections；
- actual rollout groups / completions。

因此正式实验的 raw RL budget 以后以**实际 observed rollout**为准，不用 sampler 预取数冒充计算预算。


## 正式 187-step 结果

训练代码基于 PR #9 当前分支。

固定预算：

- optimizer steps：187；
- actual rollout groups：374；
- actual RL completions：1496；
- offline screening：0；
- num_generations：4；
- exploit_fraction：0.5。

### 训练动力学

| 指标 | EXP-003 全量随机 | 005A 静态边界 | **005B 在线动态** |
|---|---:|---:|---:|
| zero-grad fraction | 27.81% | 11.76% | **10.70%** |
| zero-grad steps | 52 | 22 | **20** |
| effective steps | 135 | 165 | **167** |
| completions / effective step | 11.08 | 9.07 | **8.96** |
| mean frac_reward_zero_std | 52.14% | **33.16%** | 39.57% |
| train runtime | 751.10 s | 702.26 s | 743.26 s |

005B 没有静态 boundary 那么低的组内零方差比例，但获得了**最高的有效优化步数**，且不需要额外 1496-completion 离线筛选。

Adapter SHA-256：

`35d911cfe8f3c26fe087b44f19087ee348ff4ef94db43485bae37a8b89edcd8d`

### 在线 sampler 行为

最终状态：

- unknown：185；
- mixed：50；
- flat：139；
- unique observed tasks：189；
- exploit selections：185；
- explore selections：189；
- actual groups observed：374；
- prefetched but unobserved：0。

374 个实际 group 中：

- mixed group：226；
- flat group：148；
- observed mixed fraction：**60.43%**。

状态转移：

```text
unknown -> mixed : 84
unknown -> flat  : 105
mixed   -> mixed : 137
mixed   -> flat  : 39
flat    -> mixed : 5
flat    -> flat  : 4
```

这里最重要的是：

- `mixed -> flat = 39`；
- `flat -> mixed = 5`。

说明 task 是否位于能力边界确实依赖当前 policy，不是静态属性。

## 90 题 validation

统一设置：

- MBPP validation 90 题；
- 每题 4 候选；
- 与 EXP-003 / 005A 相同生成参数和 Docker 隐藏测试。

| 模型 | Pass@1 | Pass@4 | hidden mean | solved tasks |
|---|---:|---:|---:|---:|
| EXP-003 全量 GRPO | 27.22% | 42.22% | 29.86% | 38 |
| 005A 随机 181题 | **28.89%** | **44.44%** | **31.81%** | **40** |
| 005A 静态 boundary | 26.94% | **44.44%** | 30.28% | **40** |
| **005B 在线动态** | 28.06% | 42.22% | 31.39% | 38 |

005B 其它指标：

- syntax failures：1 / 360；
- hidden-all-pass candidates：101 / 360；
- public-all-hidden-fail：15 / 360；
- all-4-correct tasks：8。

## 配对 bootstrap（20,000 次）

### 005B - EXP-003 全量随机

Pass@1：

- observed delta：+0.83 个百分点；
- 95% CI：[-1.67, +3.61] 个百分点；
- bootstrap P(delta > 0)：0.6915。

Pass@4 solved-task：

- delta：0；
- 95% CI：[-4.44, +4.44] 个百分点。

### 005B - 005A 静态 boundary

Pass@1：

- observed delta：+1.11 个百分点；
- 95% CI：[-1.39, +3.89] 个百分点；
- P(delta > 0)：0.7626。

Pass@4：

- observed delta：-2.22 个百分点；
- 95% CI：[-7.78, +3.33] 个百分点。

### 005B - 同大小随机子集

Pass@1：-0.83 个百分点，95% CI [-3.06, +1.39]。  
Pass@4：-2.22 个百分点，95% CI [-7.78, +3.33]。

## EXP-005B 结论

### 可以确认

1. **在线策略依赖采样显著提高 rollout / 梯度利用率。**
   - zero-grad 从全量随机 27.81% 降到 10.70%；
   - 在相同 1496 completions 下多得到 32 个有效 optimizer step。

2. **边界状态确实随 policy 动态变化。**
   mixed↔flat 双向转移在正式训练中大量出现。

3. **在线方法不需要 005A 的额外离线筛选成本。**
   在有效更新效率上略优于静态 boundary。

### 不能确认

1. validation 没有出现统计上可靠的提升；
2. Pass@4 没有超过 EXP-003，并低于两个 005A 子集臂；
3. 提高“有效梯度比例”并不自动等价于扩大代码能力覆盖。

## 当前科研判断

EXP-003 → 005A → 005B 已形成一个很清楚的证据链：

```text
大量 flat group
  ↓
静态 boundary 筛选
  ↓
有效梯度显著增加
  ↓
validation 不提升
  ↓
boundary 会随 policy 变化
  ↓
在线动态采样
  ↓
有效梯度进一步提高、无需离线筛选
  ↓
validation 仍无显著提升
```

因此下一步不应该继续只优化“题目怎么采样”。

更值得进入：

- EXP-004：过程级可验证奖励 / 信用分配；
- 或更强模型 1.7B 复现实验，检查 0.6B 是否成为能力瓶颈；
- 再之后才考虑 005B-v2 的 batch 内 drop+refill。
