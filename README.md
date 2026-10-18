# CodeRL-Lab

**7B 可验证奖励后训练与能力边界实验平台**  
第一阶段稳定研究快照：`stage1-7b-v1.0.0` · 截止 2026-10-18 · 当前暂停新增训练，阶段实验已收尾。

> **阶段结论：** 已验证真实 7B GRPO 更新、部分局部程序修复及完整可追溯评测；尚未确认奖励分类修复或宽覆盖训练稳定提高外部编程能力。本仓库不是官方 EvalPlus 榜单结果，也不把负结果隐藏起来。

[阶段总报告](docs/reports/STAGE1_REPORT_ZH.md) · [复现入口](docs/reproduction/STAGE1.md) · [展示与版本说明](docs/releases/STAGE1.md) · [固定清单](releases/stage1/manifest.json) · [历史小模型 README](README.stage0-history.md)

## 从这里开始

在本稳定版本的仓库根目录，Python 3.10+ 即可运行，不需要 GPU、模型、Docker 或额外 Python 包：

```bash
python scripts/stage1_release.py demo
```

该命令**重新汇总公开的逐题计数**，不是重新生成程序或进行新的模型评测。机器可读输出：

```bash
python scripts/stage1_release.py demo --json
python scripts/stage1_release.py verify
```

安装 CPU 开发依赖后，可以复算原报告的题目配对区间并运行全部契约测试：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/stage1_release.py verify --bootstrap
python -m pytest -q
```

这里的展示命令可在普通电脑运行；项目真实 GPU 训练与容器集成记录来自 Tang，本次发布不启动任何训练。拥有 Tang 原始产物时的只读核验命令、完整重现实验需要的未随源码资产，均在[复现指南](docs/reproduction/STAGE1.md)中单独列明。

## 最后一个完整对照：固定预算下的训练题型覆盖

两臂都从同一个 Qwen2.5-7B-Instruct + TRAIN-011 SFT 起点训练，各 48 条任务、32 步、4 候选/组、256 token 上限。原有组为 6 族各 8 条；宽覆盖组为 12 族各 4 条。两个训练种子分别为 43、44。

| 模型 | 正确候选/384 | 固定首样本/96 | 四次覆盖/96 | NameError |
|---|---:|---:|---:|---:|
| SFT | 202 | 49 | 61 | 31 |
| 原有题型43 | 202 | 49 | 61 | 30 |
| 宽覆盖43 | 201 | 50 | 60 | 30 |
| 原有题型44 | 203 | 50 | 60 | 28 |
| 宽覆盖44 | 207 | 51 | 61 | 30 |

主比较“宽覆盖−原有题型”：seed43 为 **−0.26 个百分点**，名义95%区间 **[−1.30,+0.78]**；seed44 为 **+1.04 个百分点**，区间 **[−0.26,+2.60]**。区间均包含零，方向不一致。单个种子相对 SFT 的次要正结果不替代这个预设主比较。

[完整证据、资源修订与限制](docs/experiment-logs/2026-10-18-exp029-coverage-evaluation-final.md) · [可复算逐题计数](results/stage1/coverage_task_counts.csv)

## 系统做了什么

```text
固定模型 / SFT / 任务 / 代码版本
    → GPU 候选生成 → 执行前证据留存
    → 受限 Docker → 逐输入执行、依赖和语义反馈
    → GRPO 更新 → 模型及奖励日志验收
    → 全模型答案冻结 → 参考解控制检查
    → 统一资源协议评分 → 按题目配对统计与归因审计
```

本阶段着重解决四类问题：奖励是否有区分度；优化循环是否真正改变参数；局部修复与回退是什么；观察到的成绩差异能否排除执行时限混杂。格式或执行代理奖励可能饱和；依赖分类修复正确也不等于在实际轨迹中触发；已完成执行观测复用不等于新增独立样本。

## 研究进程与证据

| 阶段 | 回答的问题 | 结果入口 |
|---|---|---|
| EXP-020/021 | 三种子外部效果能否重复？ | [24题](docs/experiment-logs/2026-10-15-exp020-three-seed-mbppplus-final.md)、[新增96题](docs/experiment-logs/2026-10-17-exp021-96task-final.md) |
| EXP-022 | 哪些程序真正修复，哪些发生回退？ | [依赖反事实](docs/experiment-logs/2026-10-17-exp022-typing-counterfactual.md) |
| EXP-023 | 格式、执行代理、正确率反馈是否有效更新？ | [三臂消融](docs/experiment-logs/2026-10-17-exp023-three-arm-final.md) |
| EXP-024—027 | 严格奖励和分类修复是否有额外贡献？ | [执行混杂更正](docs/experiment-logs/2026-10-18-exp026-timeout-confounding-addendum.md)、[受控重训](docs/experiment-logs/2026-10-18-exp027-matched-execution-final.md) |
| EXP-028/029 | 固定预算扩展题型能否迁移？ | [完整覆盖结果](docs/experiment-logs/2026-10-18-exp029-coverage-evaluation-final.md) |

四批 MBPP+ 不是同一次实验；同题反复分析后不再是新盲测；旧资源协议停止记录不被成功重判覆盖。项目保存正结果、负结果和不能归因的结果。

## 稳定版本的范围

**包含：** 中文总报告、只读展示与核验命令、逐题聚合计数、版本和哈希清单、研究源码与测试、失败及纠错记录。

**不包含：** 模型权重、Tang 的候选正文和隐藏测试、完整离线模型缓存、Docker 镜像二进制、账号凭据；不承诺仅克隆仓库即可从零恢复所有历史 GPU 训练。公开计数复算、原始产物核验和重新训练是三个不同层级。

这次交付在独立发布分支整理，不合并其他训练、科研或全栈开发分支；Python 包自身版本与研究快照标签的含义不同。历史入口继续保留，但不建议为了展示项目直接运行历史自动训练脚本。

## 目录入口

```text
scripts/stage1_release.py              默认只读入口
releases/stage1/manifest.json          来源、版本、固定结果与资产哈希
results/stage1/coverage_task_counts.csv 96题五模型聚合计数
 docs/reports/STAGE1_REPORT_ZH.md       阶段总报告
 docs/reproduction/STAGE1.md           复现层级、环境、资产与失败处理
 docs/releases/STAGE1.md               展示顺序与发布边界
 docs/experiment-logs/                 按日期保存的原始阶段报告
src/coderl_lab/                        历史研究实现，不因本轮整理改变
```

## 当前决定

暂停新增训练，保留本阶段的有限结论。未来问题必须重新定义主要变量、独立验收和停止条件，不为获得正分无限追加种子、换题或调整奖励。
