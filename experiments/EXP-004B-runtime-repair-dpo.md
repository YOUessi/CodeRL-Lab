# EXP-004B：局部运行时修复偏好学习

## 状态

- 偏好对构建器：已实现第一版
- 最小 import 修复数据审计：待运行
- DPO：待数据审计通过后实现
- 正式训练：未开始

## 动机

EXP-004A 表明：

- 更细的执行阶段 reward 能显著减少 zero-grad / zero-std；
- dependency incomplete 与 runtime failure 也略有下降；
- 但 MBPP n=16 与 MBPP+ 均没有能力增益证据。

因此不再继续调 reward shaping。

EXP-006B / EXP-004A 已定位到具体 failure mode：

- 候选知道应使用 re / math / Counter / heapq 等；
- 但缺失 import，导致 NameError；
- 这类错误与算法主体无关，适合做局部偏好学习。

## v1：只做“标准库 import 最小修复”

数据来源：

- 1.7B SFT policy 的 374 train tasks × 4 rollout；
- 1496 个固定 candidates；
- EXP-004A 已生成的 dependency/runtime diagnostics。

只接受满足全部条件的 pair：

1. rejected 原候选公共测试未全通过；
2. unresolved names 全部属于标准库白名单；
3. chosen 只是在 rejected 顶部补必要 import；
4. 除 import 外不改算法主体；
5. chosen 在 train public tests 上 100% 通过；
6. hidden tests 完全不参与 pair 构建。

格式：

prompt + chosen + rejected。

## 为什么先做数据审计

很多 NameError 并不是 import 问题：

- 未定义项目 helper；
- 算法主体错误；
- re 虽缺 import，但使用方式本身也错。

这些样本不能强行变 preference pair。

因此只有“最小 import 后公共测试直接全通过”的样本才进入 v1。

## 数据门槛

进入 DPO 前至少要求：

- verified pair >= 20；
- distinct task >= 10；
- chosen/rejected 只存在 import 前缀差异；
- hidden_tests_used = false；
- 每个 chosen 都重新执行 public tests 并全通过。

若数量不足，不扩大到模糊自动修复，而先重新评估实验方向。

## verified import-repair 数据审计

来源：1.7B SFT policy 的 1496 个固定 train rollout。

严格规则：只补标准库 import，除 import 前缀外不改算法；补完后必须重新执行 public tests 并 100% 通过；hidden tests 不参与构建。

结果：

- verified pairs：97；
- distinct tasks：56；
- 每任务1对后的正式 v1 数据：56 pairs；
- pair / source prediction：6.48%；
- hidden_tests_used：false。

主要修复类型：

- import re：51；
- import math：16；
- from collections import Counter：7；
- import heapq：6；
- OrderedDict：5；
- defaultdict：3；
- statistics / groupby / deque 等少量。

被过滤：

- 无 unresolved names：1216；
- 补 import 后仍不能全过 public tests：138；
- unresolved name 非标准库白名单：22；
- duplicate pair：23。

这说明大多数 runtime failure 不是“补 import 就能修”，v1 只保留最干净的 97 个近邻样本。

正式训练数据进一步按 task 去重：每个任务只保留 original public pass rate 最高的 pair，tie 时取更小 sample_id，得到 56 对。

## 2-step DPO GPU smoke

设置：16 pair、max_steps=2、beta=0.1、sigmoid DPO、SFT adapter 为 policy 初始化，TRL 自动复制冻结 ref adapter。

结果：

- train runtime：4.01 s；
- grad norm：5.94 / 6.20，非零；
- peak allocated：4.36 GB；
- peak reserved：4.69 GB；
- adapter 保存成功；
- ref adapter 单独保存于 ref/；
- Docker 残留：0。

2-step 中 DPO reward margin 仍为 0：第1步 LR=0，第2步日志在有效更新前记录，因此 smoke 只用于工程/显存验收，不作为学习效果证据。

结论：可以进入 56 pair × 3 epoch 正式 DPO。

## 正式 DPO 训练结果

正式数据：每个任务只保留 1 个最接近正确的 verified pair，共 56 pair / 56 tasks。

训练：

- 初始化：EXP-006A 1.7B SFT adapter；
- beta=0.1；
- sigmoid DPO；
- learning rate=5e-7；
- micro batch=1；
- gradient accumulation=8；
- 3 epoch；
- 21 optimizer steps。

结果：

- train runtime：35.71 s；
- train loss：0.6734；
- peak allocated：4.57 GB；
- peak reserved：5.64 GB；
- output adapter SHA-256：860328992b8190b51ec595c08a2d36ebfecd4801db01fd8159d1bf5a0d96d18f。

### DPO 学习信号

前25% → 后25%：

- reward margin：0.0076 → 0.0579；
- preference accuracy：37.5% → 90.0%；
- DPO loss：0.6895 → 0.6649。

最后 5 个 step 的 preference accuracy 分别为 1.0、0.625、0.875、1.0、1.0。

因此 DPO 已经明确提高 chosen（最小 import 修复）相对 rejected（原 runtime-failure completion）的偏好，不是仅仅完成了训练流程。

当前 90题 × 16 validation 正在运行。最终只有 dependency/runtime 错误下降且 Pass@k 不被明显破坏，才能认为 EXP-004B v1 有效。

## v1 正式 validation 结果

90题 × 16 候选，主对照为同一 1.7B SFT 初始化。

| 指标 | SFT | Repair DPO | 变化 |
|---|---:|---:|---:|
| Pass@1 | 37.29% | 37.78% | +0.49 pp |
| Pass@4 | 54.26% | **56.24%** | +1.98 pp |
| Pass@8 | 59.68% | **63.50%** | +3.82 pp |
| Pass@16 | 63.33% | **68.89%** | +5.56 pp |
| hidden mean | 39.34% | 39.79% | +0.45 pp |
| solved@16 | 57 | **62** | +5 |
| 16/16全正确 | 9 | 10 | +1 |
| syntax failures | 4 | 7 | +3 |

配对 bootstrap（20,000次）：

- Pass@1：+0.49 pp，95% CI [-0.56,+1.60]，跨0；
- Pass@4：+1.98 pp，95% CI [+0.10,+4.07]；
- Pass@8：+3.82 pp，95% CI [+1.11,+6.95]；
- Pass@16：+5.56 pp，95% CI [+1.11,+11.11]。

因此在原 MBPP n=16 上，多样本覆盖提升具有配对统计证据。

### 但目标 failure mode 改善很小

| 诊断 | SFT | Repair DPO | 变化 |
|---|---:|---:|---:|
| dependency incomplete | 229 | 229 | 0 |
| runtime unclean | 360 | 351 | -9 |
| NameError | 239 | 236 | -3 |
| unresolved re | 77 | 74 | -3 |
| unresolved math | 46 | 48 | +2 |

所以不能把 Pass@k 提升解释为“大量 missing import 已被修好”。

### 支持集变化

SFT → DPO：

- solved@16：57 → 62；
- 新增 5 题；
- 丢失 0 题；
- Jaccard = 0.9194。

Repair DPO 与 pure GRPO 的 solved@16 都是 62，支持集 Jaccard=0.9375。
Repair DPO - pure GRPO 的 Pass@1/4/8/16 bootstrap 全部跨0。

这提示 Repair DPO 的分布行为非常接近 pure GRPO，而不只是一个局部 import 修复器。

## MBPP+ 外部强测试

39 个严格重叠任务 × 16 候选：

| 指标 | SFT | Repair DPO | 变化 |
|---|---:|---:|---:|
| Plus Pass@1 | 39.74% | 39.10% | -0.64 pp |
| Plus Pass@4 | 54.89% | 53.60% | -1.29 pp |
| Plus Pass@8 | 59.91% | 59.27% | -0.64 pp |
| Plus Pass@16 | 61.54% | 64.10% | +2.56 pp |
| solved@16 | 24 | 25 | +1 |

Plus empirical success delta：-0.64 pp，95% CI [-1.92,+0.64]，跨0。
Plus solved@16 delta：+1题，95% CI [0,+3题]，没有可靠提升证据。

因此：原 MBPP n=16 的覆盖恢复很明显，但更强 MBPP+ 没有复现同等幅度的平均成功率提升。

## v1 机制问题

当前不能判断覆盖恢复来自：

A. 最小 import 修复偏好的语义；
还是
B. 任意小规模 DPO 更新都会缓解 SFT 的概率集中。

因此在结束 EXP-004B 前必须加入随机化偏好标签控制臂。

控制原则：完全复用同 56 对 prompt/chosen/rejected 文本，只随机翻转一半 chosen/rejected 标签；训练预算、beta、学习率、epoch、seed 之外全部一致。

## 随机化偏好标签控制臂：预注册设计

目的：区分两种解释。

A. 覆盖恢复来自“最小 import 修复”的正确偏好语义；
B. 任何一次小规模 DPO 更新都能缓解 SFT 的概率集中。

控制数据完全复用正式 56 对 prompt / candidate 文本，只随机翻转 28 / 56 对的 chosen / rejected 方向。

固定：

- 文本完全相同；
- task 集完全相同；
- 56 pairs；
- 3 epoch；
- 21 optimizer steps；
- beta=0.1；
- learning rate=5e-7；
- training seed=42；
- 同一个 SFT adapter 初始化；
- label randomization seed=42042；
- 其它训练和 validation 参数完全一致。

主比较：

1. semantic repair DPO vs SFT；
2. randomized-label DPO vs SFT；
3. semantic repair DPO vs randomized-label DPO。

预注册解释标准：

- 若 semantic DPO 明显优于 random control，支持修复偏好语义本身有效；
- 若两者都相近地恢复 Pass@k，说明主要效应来自小规模 DPO 更新 / 分布解集中，而不是 import-repair 语义；
- 若 random control 更差且 semantic DPO 在 MBPP+ 仍不改善，则只能说语义偏好帮助原分布覆盖恢复，不能说提高鲁棒代码能力；
- 不因控制结果再调 beta、learning rate 或 pair 数量。

## 随机标签 DPO 控制实验

为了区分“正确 repair preference 语义”与“任意小规模 DPO 更新”的作用，构造随机标签负控制：

- 完全复用 56 对 prompt / candidate 文本；
- 28 / 56 对 chosen / rejected 随机翻转；
- label seed=42042；
- training seed=42；
- 其它超参数与语义 DPO 完全相同；
- 文本 multiset 和 prompt 逐对保持不变。

### 随机标签 DPO 训练

- 21 optimizer steps；
- runtime：32.55 s；
- train loss：0.69343；
- output adapter SHA-256：80c3b2b78d752f33348e65a3393ba9bb142588422c49599081b6e6a54e5fb186。

语义 DPO 与随机 DPO 的训练信号明显不同：

| 指标 | 语义 DPO | 随机标签 DPO |
|---|---:|---:|
| margin mean | 0.04043 | ≈0 (-0.000064) |
| margin first→last quarter | 0.0076→0.0579 | 0.0036→0.0036 |
| preference accuracy mean | 76.19% | 42.26% |
| accuracy first→last quarter | 37.5%→90.0% | 35.0%→52.5% |
| train loss | 0.6734 | 0.6934 |

因此随机标签控制确实破坏了系统性 preference 方向，没有偷偷学到与语义 DPO 相同的偏好。

## 随机标签 DPO：MBPP n=16

| 模型 | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| SFT | 37.29% | 54.26% | 59.68% | 63.33% | 57 |
| 随机标签 DPO | **38.47%** | 55.96% | 62.23% | 66.67% | 60 |
| 语义 repair DPO | 37.78% | **56.24%** | **63.50%** | **68.89%** | **62** |

### 随机标签 DPO - SFT paired bootstrap

- Pass@1：+1.18 pp，95% CI [-0.07,+2.43]；
- Pass@4：+1.71 pp，95% CI [-0.01,+3.59]；
- Pass@8：**+2.55 pp，95% CI [+0.50,+4.97]**；
- Pass@16：+3.33 pp，95% CI [0,+7.78]。

即使偏好标签随机化，仍然能恢复一部分 SFT 的多样本覆盖。

### 语义 DPO - 随机标签 DPO

- Pass@1：-0.69 pp，95% CI [-2.01,+0.56]；
- Pass@4：+0.27 pp，95% CI [-1.57,+2.22]；
- Pass@8：+1.27 pp，95% CI [-1.71,+4.43]；
- Pass@16：+2.22 pp，95% CI [-3.33,+7.78]。

全部置信区间跨 0。

所以没有统计证据证明正确 import-repair preference 语义比随机标签 DPO 更能恢复 MBPP 覆盖。

## 目标 failure mode 机制审计

对固定 validation 输出，只使用公共测试，统计“仅补标准库 import 就能让公共测试100%通过”的候选：

| 模型 | verified repairable candidates | distinct tasks | repairable / public failures |
|---|---:|---:|---:|
| SFT | 79 | 17 | 8.93% |
| 语义 DPO | 74 | 16 | 8.36% |
| 随机标签 DPO | **72** | **15** | **8.25%** |

整体 diagnostics：

| 指标 | SFT | 语义 DPO | 随机 DPO |
|---|---:|---:|---:|
| dependency incomplete | 229 | 229 | **224** |
| runtime unclean | 360 | **351** | 354 |
| NameError | 239 | 236 | **235** |
| unresolved re | 77 | **74** | 79 |

语义 DPO 并没有显著优于随机标签控制地消除它专门训练的最小 import failure；部分指标随机控制反而略好。

因此原 MBPP 的 Pass@k 恢复不能归因于“模型真正学会 import 修复”。

## MBPP+ 外部强测试：标签控制

| 模型 | Plus Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| SFT | 39.74% | 54.89% | 59.91% | 61.54% | 24 |
| 随机标签 DPO | **40.38%** | **55.69%** | **60.12%** | 61.54% | 24 |
| 语义 DPO | 38.78% | 53.06% | 59.04% | **64.10%** | **25** |

随机标签 DPO - SFT：Plus empirical success +0.64 pp，95% CI [-1.12,+2.72]，跨0。

语义 DPO - 随机标签 DPO：Plus empirical success -1.60 pp，95% CI [-4.01,+0.48]，跨0；solved@16 +1题，95% CI [0,+3题]。

外部强测试同样没有证明语义 repair preference 明显优于随机标签 DPO。

## EXP-004B v1 最终因果结论

当前证据支持以下分解：

1. **小规模 DPO 更新效应存在。**
   即使 50% 标签随机翻转、训练 margin≈0，DPO 仍能部分恢复 SFT 后的多样本覆盖。

2. **正确偏好语义确实被语义 DPO 学到了。**
   语义 DPO 的 preference accuracy 后段达到90%，margin 明显为正；随机控制没有。

3. **但“学到 import 修复语义”不是当前 Pass@k 恢复的主要因果解释。**
   语义 DPO 相对随机控制没有统计可靠的 Pass@k 优势，目标 repairable failure 也没有被更明显消除。

4. **外部 MBPP+ 没有显示语义 DPO 的平均鲁棒能力增益。**

因此更准确的研究问题变成：

> 为什么一个很小、甚至随机标签的 DPO 更新，会部分解除 SFT 后的概率集中并恢复多样本覆盖？

这个现象比继续做 import-repair 本身更值得研究。

## 下一步：EXP-004C DPO 去集中化机制

不再扩展自动修复规则。

下一阶段优先做因果控制：

1. 多个随机标签 seed，确认随机 DPO 覆盖恢复是否稳定；
2. no-op DPO（chosen == rejected）验证纯训练流水线本身不会改变模型；
3. 全反转标签 DPO，测试偏好方向是否重要；
4. continued-SFT 同预算控制，区分 DPO objective 与“任何 LoRA 更新”；
5. 对比 adapter update norm、生成熵、exact-completion diversity、Pass@k 支持集变化；
6. 再决定是否形成论文方向。
