# CodeRL-Lab

面向代码任务的**可验证奖励大语言模型后训练与能力边界实验平台**。

本项目不是“跑通一次 GRPO”的演示，而是用可执行代码任务系统研究：

- 监督微调（SFT）与强化学习（RL）究竟学到了什么不同的能力；
- 最终结果奖励与过程级可验证奖励的差异；
- 随机采样与能力边界动态采样的训练效率差异；
- 强化学习是在重新分配基础模型已有能力的概率，还是能够扩展能力边界；
- 小参数模型在不同后训练方法下的探索、熵、泛化和 Pass@k 行为。

## 当前主线

```text
基础模型
  ↓
EXP-001 基础能力评测
  ↓
EXP-002 监督微调（SFT）
  ↓
EXP-003 可验证奖励 GRPO
  ↓
EXP-005 动态采样 / 有效 rollout
  ↓
EXP-004 过程级奖励
  ↓
EXP-006 泛化 / 能力边界
```

注意：基于 EXP-003 已观测到的大量零组内方差 rollout，实验顺序调整为**先做动态采样，再做过程级奖励**。这样可以一次只改一个变量，先回答“哪些题值得 rollout”，再回答“奖励应该多细”。

第一阶段使用 Qwen3-0.6B-Base 把完整后训练链路做正确，随后再把核心实验迁移到约 1.7B 量级模型。

## 当前进度

### EXP-001：基础模型能力评测 ✅

已完成：

- 统一任务格式；
- Docker 受限代码执行；
- 公共/隐藏测试隔离；
- 可验证训练奖励；
- Pass@k；
- 多样本生成；
- 固定随机种子复现；
- Tang 4090 Laptop GPU 环境验证。

### 数据层 v1：MBPP ✅

固定数据版本：

`google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c`

分割：

- train：374；
- validation：90；
- test：500；
- prompt/few-shot：10，不进入训练。

构建结果通过逐文件 SHA-256 复现检查。

### EXP-002：监督微调基线 ✅

模型：

`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`

方法：

- LoRA；
- 374 个训练样本；
- 3 epoch；
- 只对正确代码补全部分计算损失。

90 题 validation、每题 4 次采样：

| 指标 | 基础模型 | SFT |
|---|---:|---:|
| Pass@1 | 13.61% | **25.56%** |
| Pass@4 | 38.89% | **43.33%** |
| 语法失败 | 182 / 360 | **0 / 360** |
| 平均隐藏测试通过率 | 14.86% | **28.19%** |

SFT 对 Pass@1 的提升明显大于 Pass@4。

### EXP-003：可验证奖励 GRPO ✅

初始化：EXP-002 SFT adapter。

训练：

- 374 个 train prompt；
- 每个 prompt 4 个候选；
- 纯 `loss_type="grpo"`；
- `beta=0`；
- 只使用公共测试奖励；
- 1 epoch / 187 optimizer steps。

90 题 validation：

| 指标 | Base | SFT | SFT + GRPO |
|---|---:|---:|---:|
| Pass@1 | 13.61% | 25.56% | **27.22%** |
| Pass@4 | 38.89% | **43.33%** | 42.22% |
| 语法失败 | 182 / 360 | 0 / 360 | 0 / 360 |
| 平均隐藏测试通过率 | 14.86% | 28.19% | **29.86%** |
| 至少一条正确候选的题 | 35 / 90 | **39 / 90** | 38 / 90 |
| 4/4 全正确的题 | 0 | 8 | **12** |

GRPO 的行为非常明确：

- Pass@1 相对 SFT：**+1.67 个百分点**；
- Pass@4 相对 SFT：**-1.11 个百分点**；
- 4/4 稳定成功题更多；
- 但至少能探索出一次正确答案的题反而少 1 道。

这提示当前 GRPO 更像是在进一步**集中已有成功轨迹概率**，而不是明显扩大可探索覆盖范围。

### EXP-003 训练动力学

187 个优化步：

- 平均 reward：0.4342；
- 平均 public test reward：0.3753；
- 平均 entropy：0.2770；
- 平均 `frac_reward_zero_std`：**52.14%**；
- grad norm = 0：**52 / 187 步（27.81%）**；
- 整个 batch reward std = 0：28 / 187 步。

前 25% → 后 25%：

- public reward：0.3723 → **0.4742**；
- zero-std group 比例：0.5109 → **0.5109**，没有改善。

因此项目下一步不是盲目延长 GRPO，而是研究：

> **怎样减少对“全对/全错/组内无差异”题目的无效 rollout，把预算集中到模型当前能力边界附近。**

### EXP-005A：离线能力边界筛选 ✅

SFT policy 对 374 个 train task 各采样 4 次：

- mixed：181（48.40%）
- flat：193（51.60%）

三组在相同 187 optimizer steps / 1496 RL completions 下比较：

| 指标 | 全量随机 | 181题随机控制 | 181题 Boundary |
|---|---:|---:|---:|
| zero-grad fraction | 27.81% | 29.95% | **11.76%** |
| frac reward zero std | 52.14% | 54.01% | **33.16%** |
| effective steps | 135 | 131 | **165** |
| completions / effective step | 11.08 | 11.42 | **9.07** |
| Pass@1 | 27.22% | **28.89%** | 26.94% |
| Pass@4 | 42.22% | **44.44%** | **44.44%** |
| solved tasks | 38 | **40** | **40** |

结论：

> 静态 boundary selection 显著提高 rollout / 梯度利用率，但当前 validation 没有证据显示它优于同大小随机子集。

Boundary - random 的 Pass@1 配对 bootstrap 95% CI：

`[-5.00, +0.83]` 个百分点，包含 0。

而且 005A 需要额外 1496 个 screening completions，所以不能宣称端到端更省算力。

### EXP-005B：在线策略依赖动态采样 ✅

005A 证明静态 boundary 可以显著减少无效更新，但不能改善 validation，而且需要额外离线 screening。

005B-v1 因此改为：

```text
当前 policy
  ↓
exploration slot：优先选择未观察任务
+
exploitation slot：优先选择当前 mixed 任务
  ↓
4 completions / prompt
  ↓
公共测试 reward
  ↓
实时更新 unknown / mixed / flat
  ↓
mixed↔flat 可双向转移
```

它不修改 TRL 的 reward→advantage→loss 内核，只改变 prompt 分配。

固定与 EXP-003 完全相同：

- 187 optimizer steps；
- 374 actual prompt groups；
- 1496 actual RL completions；
- 无离线 screening；
- 相同 SFT 初始化、reward、学习率和生成参数。

训练效率：

| 指标 | 全量随机 GRPO | 静态 Boundary | **在线动态** |
|---|---:|---:|---:|
| zero-grad fraction | 27.81% | 11.76% | **10.70%** |
| effective steps | 135 | 165 | **167** |
| completions / effective step | 11.08 | 9.07 | **8.96** |
| mean frac_reward_zero_std | 52.14% | **33.16%** | 39.57% |
| 额外 screening completions | 0 | 1496 | **0** |

在线状态确实持续变化：

- unknown→mixed：84
- unknown→flat：105
- mixed→mixed：137
- **mixed→flat：39**
- **flat→mixed：5**
- flat→flat：4

共实际观察 189 个不同任务；374 个 rollout group 中 60.43% 为 mixed。

90 题 validation：

| 模型 | Pass@1 | Pass@4 | hidden mean | solved tasks |
|---|---:|---:|---:|---:|
| SFT | 25.56% | 43.33% | 28.19% | 39 |
| 全量随机 GRPO | 27.22% | 42.22% | 29.86% | 38 |
| 181题随机控制 | **28.89%** | **44.44%** | **31.81%** | **40** |
| 静态 Boundary | 26.94% | **44.44%** | 30.28% | **40** |
| **在线动态** | 28.06% | 42.22% | 31.39% | 38 |

在线动态相对全量随机：

- Pass@1：+0.83 个百分点；
- Pass@4：0；
- Pass@1 配对 bootstrap 95% CI：[-1.67, +3.61] 个百分点。

因此当前证据是：

> **在线动态采样明显提高了 GRPO 的有效更新/rollout 利用率，但没有产生统计上可靠的 validation 增益。**

这说明“减少无效梯度”本身并不足以扩大模型能力。

### EXP-006A：Qwen3-1.7B 规模复现 ✅

固定模型：

`Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`

统一 90 题 validation：

| 模型 | Pass@1 | Pass@4 | hidden mean | solved tasks | 4/4全对 |
|---|---:|---:|---:|---:|---:|
| 1.7B Base | 33.61% | **64.44%** | 34.72% | **58** | 4 |
| 1.7B SFT | **36.67%** | 53.33% | **38.61%** | 48 | **19** |
| 1.7B SFT+GRPO | 36.11% | 55.56% | **38.61%** | 50 | 17 |

关键发现：

- 1.7B Base 相比 0.6B Base：Pass@1 +20.00、Pass@4 +25.56 个百分点；
- SFT 后语法失败从 92/360 降到 1/360；
- 但 SFT 的 Pass@4 从 64.44% 降到 **53.33%**；
- SFT 的 solved tasks 从 58 降到 **48**；
- 配对 bootstrap 的 Pass@4 差异 95% CI 为 **[-20.00, -2.22]** 个百分点；
- GRPO 相对 SFT 只小幅回收 Pass@4（+2.22），没有统计可靠变化。

1.7B GRPO 训练仍有明显 reward 饱和：

- mean frac_reward_zero_std：**60.16%**；
- zero-grad：**60 / 187（32.09%）**；

比 0.6B 的 52.14% / 27.81% 更严重。

因此，跨规模最稳定的两个现象是：

1. **后训练会显著改变能力分布/覆盖，而不只是提高一个准确率；**
2. **组内无差异 rollout 在更强模型上仍然大量存在。**

### EXP-006B：大 k 能力边界 / 支持集保持 ✅

固定 1.7B Base / SFT / GRPO，不再训练新模型，只扩大采样预算。

| Policy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| Base | 34.31% | **60.63%** | **65.80%** | 67.78% | 61 |
| SFT | 37.29% | 54.26% | 59.68% | 63.33% | 57 |
| GRPO | **38.82%** | 56.85% | 63.63% | **68.89%** | **62** |

关键结果：

- k=4 时看起来被 SFT 丢掉的 15 个 Base 可解任务，到了 n=16 已有 **8/15** 被 SFT 自己重新采回；
- 按 n=16 严格定义，仍有 9 个 `Base solved / SFT unsolved` 任务；
- 对这 9 题做严格 k=64 续采样，三条 policy 的前 16 个样本均与正式 n=16 **144/144 逐字节一致**；
- SFT 恢复曲线：**0/9 @16 → 4/9 @32 → 8/9 @64**；
- GRPO：**5/9 @16 → 6/9 @32 → 8/9 @64**。

唯一仍 0/64 的任务是 `mbpp_validation_0534`：

- Base 也只有 1/64 正确；
- SFT 60/64 候选会用 `re.*`，但 60/60 忘记 `import re`；
- 只补一行 `import re`，不改其它代码，**17/64** 立即通过隐藏测试；
- GRPO 对应为 **16/64**；
- 如果真实成功率等于 Base 的 1/64，那么 64 次全部失败的概率仍约 **36.5%**。

因此当前证据更支持：

> **后训练主要重新分配已有正确轨迹概率，并改变代码完整性/输出形式分布；当前没有可靠证据证明 SFT 或 GRPO 删除了 Base 的底层算法能力支持。**

### EXP-006C：MBPP+ 更强隐藏测试复核 ✅

不重新生成模型输出，直接复用 EXP-006B 的 1.7B Base / SFT / GRPO n=16 固定候选，只把测试替换为 EvalPlus MBPP+ v0.2.0。

严格同题交集 39 道，entry_point 39/39 一致；每题平均约 105 个额外测试。

| Policy | MBPP+ Pass@1 | MBPP+ Pass@4 | MBPP+ Pass@8 | MBPP+ Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| Base | 34.78% | **61.83%** | **67.56%** | **69.23%** | **27/39** |
| SFT | 39.74% | 54.89% | 59.91% | 61.54% | 24/39 |
| GRPO | **40.54%** | 54.35% | 58.51% | 61.54% | 24/39 |

结论：

- MBPP+ 会淘汰约 17%–20% 原 base tests 判定为正确的候选，说明原始 MBPP 测试确实偏弱；
- 但更强测试没有改变核心现象：SFT/GRPO 仍提高单次成功率，同时多样本覆盖低于 Base；
- GRPO 相对 SFT 的 MBPP+ Pass@1 只 +0.80 个百分点，95% bootstrap CI [-0.96, +3.04]，没有可靠提升；
- SFT/GRPO 的 plus solved@16 都是 24/39，Base 是 27/39。

所以 EXP-006A/006B 的“概率重分配 + 输出完整性重塑”并不是原始 MBPP 测试过弱造成的假象。

### EXP-004A：执行阶段可验证奖励 ✅

第一版过程奖励拆分为：

- 语法正确；
- 依赖完整性；
- 运行时干净；
- 公共测试通过率；
- 公共测试全通过奖励。

训练预算与 1.7B pure GRPO 完全匹配：187 optimizer steps / 1496 RL completions。

训练效率：

| 指标 | pure GRPO | process GRPO |
|---|---:|---:|
| zero-grad fraction | 32.09% | **18.18%** |
| mean frac_reward_zero_std | 60.16% | **43.05%** |
| effective steps | 127 | **153** |
| completions / effective step | 11.78 | **9.78** |

代码完整性：

- dependency incomplete：202 → **185**
- runtime unclean：326 → **314**
- syntax failures：8 → **4**

但 matched n=16 validation：

| 指标 | pure GRPO | process GRPO |
|---|---:|---:|
| Pass@1 | 38.82% | 39.72% |
| Pass@4 | 56.85% | 56.87% |
| Pass@8 | **63.63%** | 62.37% |
| Pass@16 | **68.89%** | 66.67% |
| solved@16 | **62** | 60 |

Pass@1/4/8/16 的 paired bootstrap 95% CI 全部跨 0。

MBPP+ 外部强测试同样没有观察到能力增益：

- Plus Pass@1：40.54% → 40.22%
- Plus Pass@4：54.35% → 54.37%
- Plus Pass@8：58.51% → 59.06%
- Plus Pass@16：61.54% → 61.54%
- solved@16：24 → 24

因此：

> **更密的执行阶段 credit signal 显著减少训练浪费，但没有自动转化成更强的最终 policy。**

### EXP-004B：局部运行时修复偏好学习（DPO） ✅

从 1.7B SFT 的 1496 个固定 train rollout 中，严格构造只补标准库 import、补完后 public tests 100% 通过的近邻偏好对：

- verified pairs：97；
- distinct tasks：56；
- 正式训练每任务只保留 1 对：56 pairs；
- hidden tests used：false。

Semantic repair DPO（21 steps）在 MBPP n=16 上：

| Policy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | solved@16 |
|---|---:|---:|---:|---:|---:|
| SFT | 37.29% | 54.26% | 59.68% | 63.33% | 57 |
| Random-label DPO | **38.47%** | 55.96% | 62.23% | 66.67% | 60 |
| Semantic repair DPO | 37.78% | **56.24%** | **63.50%** | **68.89%** | **62** |

关键控制结果：

- Random-label DPO 相对 SFT 的 Pass@8：+2.55 pp，95% CI [+0.50,+4.97]；
- Pass@16：+3.33 pp，95% CI [0,+7.78]；
- Semantic DPO 相对 Random-label DPO 的 Pass@1/4/8/16 bootstrap 区间全部跨0；
- Semantic DPO 虽然明确学会了正确 preference margin，但 dependency/NameError 改善很小；
- MBPP+ 也没有证明 semantic repair DPO 的平均成功率优于 random control。

因此当前机制证据更支持：

> **小规模 DPO 更新本身会部分缓解 SFT 的概率集中，恢复多样本覆盖；正确的 import-repair 语义不是当前覆盖恢复的主要可识别因果来源。**

### 当前优先：DPO 去集中机制

下一步不再扩大 import-repair 数据，也不继续调 beta / learning rate。

要拆开四种可能机制：

1. 成对对比目标本身是否把概率从高频模式重新分配到低概率轨迹；
2. random-label / reversed-label DPO 是否都产生类似覆盖恢复；
3. reference-policy / beta 约束是否是去集中的关键；
4. no-op / zero-gradient 控制能否排除“只是训练与重新保存 adapter 的数值扰动”。

核心目标从“修 import”转成：

> **解释为什么 DPO 能恢复 SFT 在多样本采样下丢失的覆盖，以及这种恢复是否可预测、可控制。**

## 快速开始

CPU 开发环境：

```bash
python -m pip install -e ".[dev]"
pytest -q
```

准备固定 MBPP-v1：

```bash
python -m pip install -e ".[data]"
bash scripts/prepare_mbpp.sh
```

模型环境：

```bash
python -m pip install -e ".[dev,model]"
```

运行 EXP-002 SFT：

```bash
bash scripts/run_sft_exp002.sh
bash scripts/run_exp002_eval.sh
```

运行 EXP-003 GRPO：

```bash
bash scripts/run_grpo_exp003.sh
bash scripts/run_exp003_eval.sh
```

模型生成代码默认必须通过 Docker 执行。仓库内可信夹具才允许显式使用本地执行模式。

## 开发与实验工作流

**GitHub 是唯一事实源（source of truth）**。

所有代码、配置、实验说明、实验记录、结果摘要都先提交到 GitHub。  
Tang（RTX 4090 Laptop GPU，16 GB）仅作为 GPU 执行节点：需要 CUDA、模型推理、训练或 benchmark 时，拉取指定 GitHub 分支/提交运行。

每个实验必须绑定：

- 实验编号；
- Git 分支；
- Commit SHA；
- 模型与数据版本；
- 完整配置；
- 随机种子；
- 硬件环境；
- 关键指标；
- 异常与失败记录；
- 结论与下一步。

## 实验原则

1. 隐藏测试永远不进入训练奖励。
2. 不仅报告 Pass@1，同时报告更大的 Pass@k。
3. 失败实验同样记录。
4. 大模型权重、checkpoint、缓存、完整 rollout 不进入 Git。
5. 配置、结果、实验日志、关键图表必须进入 Git。
6. 每个结论必须追溯到代码提交和配置。
7. MBPP 不能单独承担“无污染能力边界”结论。
8. 最终能力边界结论必须加入 MBPP+ / LiveCodeBench 等外部评测。
9. 算法对照一次尽量只改变一个变量。

## 路线图

- [x] EXP-001：基础模型 Pass@k 基线
- [x] 数据层 v1：固定 MBPP 训练 / 验证 / 测试分割
- [x] EXP-002：监督微调基线
- [x] EXP-003：纯 GRPO + 最终结果奖励
- [x] EXP-005A：离线能力边界筛选 + 同大小随机控制
- [x] EXP-005B：在线策略依赖动态采样
- [x] EXP-004A：执行阶段可验证奖励
- [x] EXP-004B：局部运行时修复偏好学习
- [x] EXP-006A：1.7B Base / SFT / GRPO 规模复现
- [x] EXP-006B：大 k 能力边界 / 支持集保持
- [ ] EXP-007：奖励投机与隐藏测试鲁棒性
- [x] MBPP+ 外部强测试复核
- [ ] LiveCodeBench 时间更新外部分布评测
- [x] 1.7B 主模型迁移
- [ ] 仓库级软件工程任务扩展
