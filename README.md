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

### 当前优先：EXP-005B 在线动态采样

005A 直接暴露了静态筛选的限制：

> 一个题在初始 SFT policy 下是 mixed，不代表训练若干步后仍然是 mixed。

因此 005B 改为 policy-dependent 在线策略：

```text
当前 policy
  ↓
采样 prompt group
  ↓
计算公共测试 reward
  ↓
reward spread = 0 ?
  ├─ 是：丢弃该 group，补采新的 prompt
  └─ 否：进入 GRPO update
```

核心对照要分两个预算口径：

1. **固定 raw rollout budget**：看动态采样能否用相同生成成本获得更多有效更新；
2. **固定 effective-update budget**：看为了获得同样有效更新，动态采样需要多少总 rollout。

005B 还会保留随机/多样性成分，避免只围绕少量边界题反复训练。

EXP-004 过程奖励继续作为后续独立变量，不与采样策略同时修改。

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
- [ ] EXP-005B：在线动态采样（当前优先）
- [ ] EXP-004：过程级可验证奖励
- [ ] EXP-006：SFT 与 RL 泛化 / 能力边界对照
- [ ] EXP-007：奖励投机与隐藏测试鲁棒性
- [ ] MBPP+ / LiveCodeBench 外部评测
- [ ] 1.7B 主模型迁移
- [ ] 仓库级软件工程任务扩展
