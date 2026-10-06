# EXP-006B：大 k 能力边界 / 支持集保持

## 状态

**设计：完成。**  
**16-sample GPU smoke：已通过。**  
**90题 × 16 正式采样：已完成。**

## 研究动机

EXP-006A 在 Qwen3-1.7B 上观察到：

| 模型 | Pass@1 | Pass@4 | solved tasks |
|---|---:|---:|---:|
| Base | 33.61% | **64.44%** | **58** |
| SFT | **36.67%** | 53.33% | 48 |
| GRPO | 36.11% | 55.56% | 50 |

其中 SFT - Base 的 Pass@4 配对 bootstrap 95% CI：

`[-20.00, -2.22]` 个百分点。

但 k=4 仍不能区分：

1. SFT 只是把 Base 原有正确轨迹概率压低，因此 4 次没采到；
2. 后训练后这些正确轨迹在经验可采样支持中真的显著减少。

EXP-006B 不训练新模型，只扩大采样预算。

## 固定模型

三条 policy 全部固定，不再更新：

### Base

`Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`

### SFT

adapter SHA-256：

`e1ea9a007a3e4213cdeb1ca6b1e12d29adb83594991421757d0d40b9168d0861`

### GRPO

adapter SHA-256：

`b858f377e43a7cb75ebd9ef332d2f77dba9728d25954fadc4e512a26a47da46c`

## 固定评测

- MBPP validation 90 题；
- 每题统一生成 n=16；
- temperature=0.8；
- top-p=0.95；
- max_new_tokens=512；
- seed=42；
- 相同 prompt；
- 相同代码归一化；
- 相同 Docker hidden tests。

计算：

- Pass@1
- Pass@4
- Pass@8
- Pass@16

同时直接分析 n=16 经验支持集：

- 至少 1/16 正确的任务；
- Base → SFT 保留率；
- Base 可解但 SFT 0/16 的任务；
- SFT 新获得任务；
- GRPO 恢复多少 SFT 丢失的 Base 任务；
- Base/SFT/GRPO 支持集 Jaccard。

## 第一阶段 smoke

validation 前 2 题：

```text
3 policies × 2 tasks × 16 samples
```

目标：

- 验证一次返回 16 sequences 不 OOM；
- 记录峰值显存；
- 验证 sample_id 0..15；
- 验证 Pass@1/4/8/16 评测；
- 三条 policy 都使用同一生成方式。

如果 16-return batch 在 16GB 上 OOM，再只调整生成分块，不改变采样总数和模型。

## 正式结论判据

### 更支持“概率重分配”

如果：

- SFT Pass@4 明显低于 Base；
- 但 Pass@16 / solved@16 基本恢复；
- Base 在 k=4 丢失的题，大部分在 SFT 16 次中重新出现；

则更像：

> 正确轨迹仍存在，只是概率下降。

### 更支持“经验支持集收缩”

如果：

- 到 k=16 SFT solved tasks 仍明显低于 Base；
- 大量 Base 16次可解任务在 SFT 16次仍 0；
- GRPO 也难恢复；

则说明：

> 后训练可能在当前可观察采样预算下造成能力支持集收缩。

这仍不是数学意义的“能力消失”，最终需要更大的 k 和外部基准验证。

## 第二阶段：针对性 k=64

不直接对全部 90 题跑 64。

若 n=16 后仍存在稳定的：

`Base solved / SFT unsolved`

任务，则只对这批关键任务继续采样到 k=64，降低计算浪费。

## 当前边界

MBPP 是公开老基准，EXP-006B 只能研究**同一基础模型后训练前后的相对支持变化**。

最终无污染能力边界结论仍需要 LiveCodeBench / 更近期任务。


## 16-sample GPU smoke 结果

代码提交：

`6ecc3ad42dd72dc7135ca054368857d24f999e28`

validation 前 2 题，三条 policy 各 16 samples。

### 显存

| Policy | peak allocated | peak reserved |
|---|---:|---:|
| Base | 4,662,031,360 | **5,937,037,312** |
| SFT | 4,046,539,776 | 4,221,566,976 |
| GRPO | 4,003,616,768 | 4,190,109,696 |

16-return generation 在 Tang 16GB 上稳定，无需分块。

### Smoke 功能验证

- sample_id 0..15 正常；
- Base/SFT/GRPO 都完成 32 个候选；
- Pass@1/4/8/16 正常计算；
- 支持集分析器正常输出保留/新增/恢复任务；
- Docker 隐藏测试正常；
- 无 OOM。

2 题 smoke 的数值不用于科研结论，只作为工程门槛。

因此 EXP-006B 进入完整 90题 × 16 候选实验。


## 正式运行的完整性保护

正式 n=16 实验增加两层保护：

### 1. 三模型 prediction 网格完整性

在最终分析前强制验证：

- Base：90 × 16 = 1440 rows；
- SFT：90 × 16 = 1440 rows；
- GRPO：90 × 16 = 1440 rows；
- 每个 task 的 sample_id 必须严格为 0..15；
- 三组 task_id 集完全一致；
- 不允许旧评测产物在缺少本轮 predictions 时进入分析。

### 2. 针对性 k=64 严格续采样

如果 n=16 后仍存在 `Base solved / SFT unsolved` 任务，不直接用抽出的子集重新设 seed。

原因：当前生成器的 task seed 与原 validation 序号相关；抽子集会改变序号，导致所谓“17–64”并不是对原 1–16 的继续。

因此新增：

- 从正式 n=16 sample_id=0 记录中恢复每题原 task seed；
- 对目标任务用原 seed map；
- 按 16 个一块继续生成到 64；
- 强制比较新 n=64 文件中 sample 0..15 与正式 n=16 原文件；
- 任一 completion 不一致则停止，不允许进入 k=64 结论。

这样 k=64 才是对 n=16 的严格扩展，而不是另一轮随机重采样。


# n=16 正式结果

三条固定 policy 均完成：

```text
90 tasks × 16 samples = 1440 completions / policy
3 policies = 4320 completions
```

## Pass@k

| Policy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| Base | 34.31% | **60.63%** | **65.80%** | 67.78% | 61/90 |
| SFT | 37.29% | 54.26% | 59.68% | 63.33% | 57/90 |
| GRPO | **38.82%** | 56.85% | 63.63% | **68.89%** | **62/90** |

### 语法与执行

| Policy | 语法失败 | hidden mean | hidden-all-pass candidates |
|---|---:|---:|---:|
| Base | 352 / 1440 | 36.32% | 494 |
| SFT | **4 / 1440** | 39.34% | 537 |
| GRPO | 8 / 1440 | **41.32%** | **559** |

## n=16 经验支持集

### Base → SFT

- Base solved：61；
- SFT solved：57；
- Base retained by SFT：52 / 61 = **85.25%**；
- Base solved / SFT unsolved：**9**；
- SFT 新增 vs Base：5；
- Jaccard(Base, SFT)：0.7879。

因此，k=4 时看起来非常严重的覆盖下降，在 n=16 下已经明显缩小，但没有完全消失。

### SFT → GRPO

- SFT solved：57；
- GRPO solved：62；
- GRPO 新增 vs SFT：6；
- SFT lost by GRPO：1；
- Jaccard(SFT, GRPO)：0.8889。

在 Base 被 SFT 丢掉的 9 个 n=16 任务中，GRPO 恢复 5 个：

```text
recovery_fraction = 5 / 9 = 55.56%
```

## 追踪 EXP-006A 的 15 个 k=4 丢失任务

EXP-006A 中共有 15 题满足：

`Base k=4 solved / SFT k=4 unsolved`

到了 n=16：

- SFT 恢复：**8 / 15（53.33%）**；
- SFT 仍 0/16：7；
- GRPO 在这 7 题中又恢复：3。

这直接说明：

> k=4 下超过一半的所谓“能力丢失”不是经验支持集彻底消失，而是正确轨迹概率下降后 4 次采样没有命中。

## 配对 bootstrap（n=16）

### SFT - Base：经验成功率

- delta：+2.99 个百分点；
- 95% CI：[-1.94, +7.99]；
- P(delta > 0)：0.8804。

### SFT - Base：solved@16

- delta：-4.44 个百分点；
- 95% CI：[-12.22, +3.33]；
- P(delta > 0)：0.1131。

到 n=16 后，SFT 的覆盖下降不再有 k=4 时那样明确的统计证据。

### GRPO - SFT：经验成功率

- delta：**+1.53 个百分点**；
- 95% CI：**[+0.21, +2.85]**；
- P(delta > 0)：0.9873。

### GRPO - SFT：solved@16

- delta：+5.56 个百分点；
- 95% CI：[0, +11.11]；
- P(delta > 0)：0.9656。

这是目前第一次在更大采样预算下看到 GRPO 相对 SFT 的经验成功率改善具有正向 bootstrap 区间。

## n=16 当前解释

现在更支持：

```text
SFT
主要改变正确轨迹概率分布
而不是简单删除全部 Base 能力
```

但仍有 9 个任务在当前 n=16 经验支持下表现为：

`Base solved / SFT unsolved`

所以第二阶段继续只针对这 9 题做严格 k=64 续采样。


# 针对性 k=64 严格续采样结果

n=16 后严格目标：9 道 `Base solved / SFT unsolved` 任务。

三条 policy 均完成 9 × 64 = 576 completions，并且：

- Base 前16：144 / 144 completion 与正式 n=16 逐字节一致；
- SFT 前16：144 / 144 一致；
- GRPO 前16：144 / 144 一致；
- missing = 0；
- mismatch = 0。

因此这是严格的 n=16 → n=64 续采样。

## 恢复曲线

### Base

| k | solved / 9 |
|---|---:|
| 16 | 9 |
| 32 | 9 |
| 64 | 9 |

### SFT

| k | solved / 9 |
|---|---:|
| 16 | 0 |
| 32 | 4 |
| 64 | **8** |

### GRPO

| k | solved / 9 |
|---|---:|
| 16 | 5 |
| 32 | 6 |
| 64 | **8** |

所以：

```text
SFT support recovery:
0/9 @16
→ 4/9 @32
→ 8/9 @64
```

这非常强地支持“正确轨迹概率下降”而不是“能力彻底消失”。

## 目标子集 Pass@k

注意：这是按 Base-solved/SFT-unsolved 选择出来的条件子集，不能作为总体模型性能比较，只用于支持恢复分析。

| Policy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | Pass@32 | Pass@64 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 21.35% | 53.02% | 68.56% | 80.17% | 90.39% | **100%** |
| SFT | 4.34% | 16.09% | 29.18% | 48.66% | 71.08% | **88.89%** |
| GRPO | 5.21% | 18.67% | 32.61% | 51.40% | 71.24% | **88.89%** |

## 唯一剩余任务：mbpp_validation_0534

任务要求：在字符串中搜索 literal，并返回匹配位置 `(start, end)`。

64 次：

- Base：1 / 64；
- SFT：0 / 64；
- GRPO：0 / 64。

Base 唯一正确样本显式包含：

```python
import re
...
match = re.search(pattern, text)
return match.span()
```

而 SFT / GRPO 的主要失败模式不是“不知道用正则”：

### SFT

- 64 个候选；
- 60 个使用了 `re.*`；
- **60 个全部缺少 `import re`**；
- 只补一行 `import re`，其它代码完全不改：**17 / 64** 立即通过隐藏测试。

### GRPO

- 64 个候选；
- 60 个使用了 `re.*`；
- 59 个缺少 `import re`；
- 只补 `import re`：**16 / 64** 立即通过隐藏测试。

因此最后这道 0/64 更像：

> 后训练把“输出完整依赖声明”的概率压低，而不是把字符串匹配算法能力删除。

## 尾部统计边界

Base 在该题只有 1/64 正确，经验成功率：

`p_hat = 1.5625%`

如果 SFT 真正成功概率仍等于这个经验值，那么 64 次全部失败的概率：

```text
(1 - 1/64)^64 ≈ 36.50%
```

因此观察到 SFT / GRPO 的 0/64 并不罕见。

对 1/64 vs 0/64 的 Fisher exact 双侧比较 p=1.0，没有证据认为两者支持概率显著不同。

## EXP-006B 最终结论

从 k=4 → 16 → 64：

1. k=4 时 SFT 看起来丢失 15 个 Base 可解任务；
2. 到 n=16，15 个中已有 8 个恢复；
3. 按 n=16 重新严格定义，剩 9 个 Base-solved/SFT-unsolved；
4. 到 k=32，SFT 恢复 4 / 9；
5. 到 k=64，SFT 恢复 **8 / 9**；
6. 唯一剩余 1 题又可由单纯补 `import re` 把 17 个 SFT 候选恢复为正确。

所以当前最符合证据的解释是：

> **后训练主要重新分配已有能力轨迹的概率，并改变输出形式/代码完整性分布；当前没有可靠证据证明 SFT 或 GRPO 删除了 Base 的底层算法能力支持。**

这不等于“后训练绝不遗忘”。它只说明在当前 MBPP、Qwen3-1.7B、LoRA SFT + GRPO、k≤64 的实验范围内，所谓能力丢失大部分可以由采样概率下降和输出完整性错误解释。

## 下一研究问题

EXP-006B 已经把“支持集是否真的缩小”推进到相当明确的结论。

下一步不应该继续无脑增大 k，而应该研究：

1. **为什么 SFT 会把完整代码依赖（例如 imports）的概率压低？**
2. **过程级/执行级奖励能否改善这种代码完整性，同时不牺牲覆盖？**
3. **这些现象能否在 MBPP+ / LiveCodeBench 等更强外部评测复现？**

这为 EXP-004 过程级可验证奖励 / 信用分配提供了直接实验动机。
