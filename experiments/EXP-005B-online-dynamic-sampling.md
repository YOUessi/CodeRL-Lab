# EXP-005B：在线策略依赖的动态采样

## 状态

**设计：完成。**  
**代码：第一版完成。**  
**GPU 冒烟：已通过。**  
**正式实验：待运行。**

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
