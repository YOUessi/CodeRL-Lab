# EXP-006B：大 k 能力边界 / 支持集保持

## 状态

**设计：完成。**  
**16-sample GPU smoke：已通过。**  
**90题 × 16 正式采样：待运行。**

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
