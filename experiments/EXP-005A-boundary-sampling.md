# EXP-005A：离线能力边界筛选 + 匹配预算 GRPO

## 状态

**代码实现：第一版完成。**  
**SFT 策略全训练集筛选：已完成。**  
**匹配预算 GRPO：待运行。**

## 为什么现在做这个实验

EXP-003 观察到：

- 平均 `frac_reward_zero_std = 52.14%`；
- 187 个优化步中 52 步 grad norm = 0；
- Pass@1 上升，但 Pass@4 小幅下降。

这说明大量 rollout 花在：

- 当前模型 4 次都成功；
- 当前模型 4 次都失败；
- 或其它组内奖励完全一样的题。

对纯 GRPO 来说，这些 prompt 的组内相对优势为 0，不能产生有效策略梯度。

因此先单独研究**样本分配**，不同时修改奖励函数。

## 研究问题

> 在相同 RL rollout / optimizer-step 预算下，只训练初始 SFT policy “有时会、有时不会”的题，能否减少无效更新并改善 validation？

## 为什么叫 005A，而不是宣称完整 DAPO 动态采样

这一版采用：

1. 用固定 SFT policy 对全部 train task 做一次离线 4-sample probe；
2. 只保留 reward spread > 0 的 `mixed` task；
3. 用选中 task 进行 187 个 GRPO update。

它是**离线能力边界课程（offline boundary curriculum）**。

真正 DAPO 式在线动态采样会随着当前 policy 变化不断重新筛题，属于后续 005B。

## 筛选策略

对每个 train task：

```text
SFT policy
  ↓
4 completions
  ↓
public-test reward
  ↓
r1, r2, r3, r4
```

分类：

- `mixed`：max(r) - min(r) > 0；
- `flat_all_pass`：4 个 reward 都为 1；
- `flat_test_fail`：语法都正确，但 4 个公共测试通过率都为 0；
- `flat_syntax_fail`：4 个都语法失败；
- `flat_partial`：其它相同奖励。

只有 `mixed` 进入 005A GRPO 训练集。

生成的 boundary training 文件**不包含 hidden_tests 字段**。

## 公平性 / 预算

### EXP-003 随机全量 GRPO

- 187 optimizer steps；
- 每步 2 个 prompt group；
- 每组 4 completions；
- RL prompt groups = 374；
- RL completions = 1496。

### EXP-005A

固定完全相同：

- 187 optimizer steps；
- 每步 2 个 prompt group；
- 每组 4 completions；
- RL prompt groups = 374；
- RL completions = 1496。

如果 mixed task 少于 374，会在训练过程中重复采样这些边界 task。

### 额外筛选成本

离线 screening 额外需要：

```text
374 × 4 = 1496 completions
```

这个成本**单独报告**。

因此：

- “RL 训练预算”与 EXP-003 匹配；
- “端到端总预算”不匹配，005A 多了 screening 开销。

不能把 005A 直接宣传成端到端更省算力。

## 同大小随机子集控制

为了排除“边界子集更小、所以同样 187 步会重复更多题”这一混杂因素，005A 增加第二个控制臂：

### 随机子集重复 GRPO

筛选得到 mixed task 数量为 M 后：

- 从全部 374 个 train task 中用 seed=42 随机抽取 M 个；
- 同样移除 hidden tests；
- 同样训练 187 optimizer steps；
- 同样 1496 个 RL completions；
- 同样从 EXP-002 SFT adapter 初始化。

于是正式对照变成：

```text
A. EXP-003：374题全量，每题大体一次
B. EXP-005A-Control：随机 M 题，重复到 187 steps
C. EXP-005A-Boundary：边界 M 题，重复到 187 steps
```

B vs C 才是判断“边界选择本身”是否有效的核心对照。

## 保持不变的变量

- Base model revision；
- SFT adapter；
- reward function；
- public tests；
- num_generations = 4；
- temperature / top-p；
- loss_type = grpo；
- beta = 0；
- learning rate；
- batch / gradient accumulation；
- 187 optimizer steps；
- validation 任务和采样配置。

唯一核心变量：

```text
随机/全量 prompt 分配
vs
SFT-policy 边界 prompt 分配
```

## 三组预算

### A. EXP-003 全量随机

- unique tasks：374
- optimizer steps：187
- RL prompt groups：374
- RL completions：1496

### B. 同大小随机子集

- unique tasks：M
- optimizer steps：187
- RL prompt groups：374
- RL completions：1496

### C. 边界子集

- unique tasks：M
- optimizer steps：187
- RL prompt groups：374
- RL completions：1496

B/C 的任务重复频率相同，只改变“随机选还是边界选”。

边界筛选额外 1496 completions 单独计入总成本。

## 主要指标

训练：

- frac_reward_zero_std；
- zero-grad step fraction；
- reward / reward std；
- entropy；
- step time；
- 每个有效梯度 update 消耗的 RL completions。

验证：

- Pass@1；
- Pass@4；
- solved tasks；
- 4/4 全正确任务；
- hidden test mean；
- 相对 EXP-003 的逐题 gained / lost。

## 成功判据

最重要的不是单纯 Pass@1 更高。

至少需要看到以下之一：

1. 相同 RL budget 下 zero-std / zero-grad 明显下降，并且 validation 不退化；
2. Pass@1 / Pass@4 至少一个明显改善，另一个不显著退化；
3. 覆盖题数增加，而不是仅进一步集中概率；
4. 为后续在线动态采样提供明确数据依据。

若只减少 zero-grad、但最终性能不变，也仍是有价值的负结果。


## SFT 策略全训练集筛选结果

筛选代码提交：

`8650cb38104c90171dae253fc3c92e3e4a42ea7f`

固定 SFT policy 对全部 374 个 train task 各采样 4 个候选，共：

```text
374 × 4 = 1496 completions
```

筛选结果：

| 类别 | 任务数 | 占比 |
|---|---:|---:|
| mixed | **181** | **48.40%** |
| flat_test_fail | 132 | 35.29% |
| flat_all_pass | 59 | 15.78% |
| flat_partial | 2 | 0.53% |
| 合计 | 374 | 100% |

因此：

- 边界训练集大小 M = **181**；
- 同大小随机控制集也固定为 **181**；
- 两组正式 GRPO 都仍跑 187 optimizer steps；
- 两组正式 RL rollout budget 都是 1496 completions；
- 边界筛选额外产生的 1496 completions 单独计费。

筛选文件哈希：

- 原 train tasks：`52f92e7d9e38c91fa25c283dcea53a507f3ea0983d6d82b9c3f91fa2a72ce689`
- predictions：`9d014c1574df072cb9f2177d6c37b91758b7fcc4375b2fbc2c0620aa4e56b75d`
- screening：`54042d35efad2d8c6232e5e33a886a5a40a8408d755bd495a5a2fcb8c76b7e06`
- boundary tasks：`27dd268487c219413ec2966595704bc6d84ed62e953645d4c2c0f974e21d2e7f`
- random control tasks：`13ca896810d9ccc82d437eae9dc95d41b0a2ac76077265bcfa0433275a01b559`

跨 374 题的平均 reward：0.4419。  
平均 reward spread：0.3636。

### 与 EXP-003 的直接对应

EXP-003 在线训练中：

`mean frac_reward_zero_std = 52.14%`

离线 SFT 筛选中非 mixed 比例：

```text
(132 + 59 + 2) / 374 = 51.60%
```

两者非常接近，说明 EXP-003 中大量零组内方差信号不是偶然 batch 现象，而是训练题在当前 policy 下的系统性难度分层。


## 同大小随机控制集组成

181 个随机控制任务（seed=42）在 SFT 筛选时原类别为：

| 类别 | 任务数 | 占比 |
|---|---:|---:|
| mixed | 93 | 51.38% |
| flat_test_fail | 57 | 31.49% |
| flat_all_pass | 30 | 16.57% |
| flat_partial | 1 | 0.55% |

因此随机控制臂仍保留了约一半 flat task；boundary 臂则是 181 / 181 mixed。

这使 B vs C 的差异非常直接：

- B：相同子集大小，但普通随机组成；
- C：相同子集大小，100% 位于初始 SFT policy 的组内奖励边界。
