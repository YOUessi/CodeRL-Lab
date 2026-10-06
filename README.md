# CodeRL-Lab

面向代码任务的**可验证奖励大语言模型后训练与能力边界实验平台**。

本项目不是“跑通一次 GRPO”的演示，而是用可执行代码任务系统研究：

- 监督微调（SFT）与强化学习（RL）究竟学到了什么不同的能力；
- 最终结果奖励与过程级可验证奖励的差异；
- 随机采样与能力边界动态采样的训练效率差异；
- 强化学习是在重新分配基础模型已有能力的概率，还是能够扩展能力边界；
- 小参数模型在不同后训练方法下的探索、熵、泛化和 Pass@k 行为。

## 研究主线

```text
基础模型
  ↓
基础能力评测
  ↓
监督微调（SFT）
  ↓
可验证奖励
  ↓
组相对策略优化（GRPO）
  ↓
过程奖励 / 动态采样
  ↓
Pass@k / 熵 / 泛化 / 能力边界分析
```

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

SFT 对 Pass@1 的提升明显大于 Pass@4，说明后续必须继续研究“概率重分配”与“能力边界”的区别。

### 当前：EXP-003

下一步进入：

```text
SFT 模型
  ↓
train split 公共测试
  ↓
可验证奖励
  ↓
GRPO
  ↓
Base / SFT / SFT+GRPO 统一评测
```

隐藏测试不会进入训练奖励。

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

模型训练环境：

```bash
python -m pip install -e ".[dev,model]"
```

运行 EXP-002 SFT：

```bash
bash scripts/run_sft_exp002.sh
bash scripts/run_exp002_eval.sh
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

## 项目结构

```text
CodeRL-Lab/
├── configs/
├── src/coderl_lab/
│   ├── datasets/
│   └── train/
├── tests/
├── benchmarks/
├── data/
│   └── manifests/
├── experiments/
├── results/
├── docs/
└── scripts/
```

## 实验原则

1. 隐藏测试永远不进入训练奖励。
2. 不仅报告 Pass@1，同时报告更大的 Pass@k。
3. 失败实验同样记录，不删除“不好看”的结果。
4. 大模型权重、检查点、缓存、完整 rollout 不进入 Git。
5. 配置、结果表、实验日志、关键图表必须进入 Git。
6. 每个结论必须能够追溯到确定的代码提交和配置。
7. MBPP 是长期公开数据，不能单独承担“无污染能力边界”结论。
8. 最终能力边界结论必须加入 MBPP+ / LiveCodeBench 等更强外部评测。

## 路线图

- [x] EXP-001：基础模型 Pass@k 基线
- [x] 数据层 v1：固定 MBPP 训练 / 验证 / 测试分割
- [x] EXP-002：监督微调基线
- [ ] EXP-003：GRPO + 最终结果奖励
- [ ] EXP-004：过程级可验证奖励
- [ ] EXP-005：能力边界动态采样
- [ ] EXP-006：SFT 与 RL 泛化 / 能力边界对照
- [ ] EXP-007：奖励投机与隐藏测试鲁棒性
- [ ] MBPP+ / LiveCodeBench 外部评测
- [ ] 仓库级软件工程任务扩展
