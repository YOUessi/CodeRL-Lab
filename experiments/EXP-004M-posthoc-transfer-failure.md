# EXP-004M：为什么 Verifier-Gated Token Escape 没有迁移到 LiveCodeBench？

## 状态及研究边界

- 日期：2026-10-08
- 分支：`exp/livecodebench-verifier-gated`
- 类型：**EXP-004K/L 正式结果后的后验诊断**，不是新独立实验、不是事先预注册的因果检验。
- 输入：已冻结的 `results/exp004k/summary.json`、`results/exp004l/summary.json`，以及 EXP-004L 在 Tang 保存的 `runner.json` / 最终正确率详情。无需任何新生成、GPU 训练或读取测试内容。
- 冻结校验：L runner SHA-256 = `506ebd7eb08519a44d2988dc76dd855a43c7ccd454059d131e97079375c3879a`。
- 禁止：在 LCB v6 上调 low/high 阈值、bias、采样温度、输入 prompt 或截断长度后称为原 EXP-004L 的成功复制。

## 原问题及四个彼此不同的故障层

1. **Gate 是否能选择真实的错误轨迹？** 观察 Gate precision / recall。
2. **干预实际是否被应用、是否改变轨迹？** 观察 applied/change fraction。
3. **新轨迹是否更有可能正确？** 观察 wrong→correct、correct→wrong，区别于 raw exact-match 变化。
4. **是否存在完成输出或模型解题能力的地板？** 检查原始 Base/SFT correctness、困难任务比例、512 token cap 和 AST 语法；这些都只是候选解释，不是先验因果结论。

## 结果 A：Gate 的错题识别仍然有效，但 rescue 出现数量级下降

| 指标 | MBPP test K (500) | LiveCodeBench v6 L (175) |
|---|---:|---:|
| SFT baseline correct | 207/500 (41.40%) | 14/175 (8.00%) |
| Gate 触发 | 159/500 (31.80%) | 148/175 (84.57%) |
| Gate 选中的确为错误题 | 152/159 (95.60%) | 148/148 (100%) |
| wrong→correct | 10 | 1 |
| correct→wrong | 2 | 0 |
| **rescue / 门控选中错误题** | **10/152 = 6.58%** | **1/148 = 0.68%** |
| 正式净准确率变化 | +1.60 pp | +0.57 pp |

上述 **rescue / 真实错误题** 是后验分层结果，不是算法自身可观测的在线指标。

探索性 Fisher exact 双侧 p = 0.01039723948（未校正多重比较）。两组数据的任务难度、题型、提示格式和执行语义都不同，不能将这个 p 值解释成经过随机控制的“分布变迁导致干预机制下降”的因果证明。

审计：`src/coderl_lab/analysis/exp004m_transfer_diagnostics.py`，机器可读：`results/exp004m/transfer_summary.json`；[GitHub Actions 实际执行](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37727689371)。

## 结果 B：不是简单“改不动轨迹”

对 L 已冻结的175题 `runner.json` 用单独哈希校验和真实正确率详情进行只读对齐：

- 有 low bottleneck 的题：159/175；
- public-fail + low bottleneck Gate：148/175；
- **实际实施干预：148/148**；
- **完整 raw 轨迹发生改变：119/148 = 80.41%**；
- 119条已改变轨迹中只有1条最终从 wrong→correct（0.84%）；
- gated-high 的完整轨迹变化为0；
- LeetCode：54题触发，43题轨迹改变，0题修复；
- AtCoder：94题触发，76题轨迹改变，1题修复；
- Hard：77题触发，60题轨迹改变，0题修复。

因此，`bias=-0.25` 在已触发的绝大多数位置确实改变了推理轨迹；真正缺失的是**改变后能通向正确程序的轨迹质量**，不能把问题仅解释为没有触发或模型对干预免疫。

审计：`src/coderl_lab/analysis/exp004m_local_trace_audit.py`，聚合产物：`results/exp004m/trace_audit_summary.json`（不包含逐题代码、private I/O 或原始测试）。

## 结果 C：512 Token 限制和语法截断风险

使用相同的冻结 runner，在**不重新解码、不执行私有测试**的条件下计算：

| 分层 | 总题数 | 达到512 Token cap | SFT 原始代码 AST 无法解析 |
|---|---:|---:|---:|
| 全部 | 175 | 41 | 28 |
| Easy | 43 | 3 | 2 |
| Medium | 52 | 6 | 5 |
| Hard | 80 | 32 | 21 |

- 原始 SFT：147/175 可以通过 Python `ast.parse`；干预后：143/175；
- **28 个原始语法失败都属于达到512-token cap 的任务**；
- Hard 80题中32题达到 cap、21题无法语法解析；Hard 最终仍为0/80正确；
- 但有147题语法可解析也只有14题原始正确，因此“只修生成长度”不足以解释所有失败。

> 达到512-token cap 只是“可能截断”的代理指标，不等于自动证明截断，更不能说明所有错误都因此产生。AST可解析也不等于功能正确。提高 cap 是新的实验条件，不能回头改写原本冻结的004L。

审计：`src/coderl_lab/analysis/exp004m_trace_quality.py`、`results/exp004m/quality_audit_summary.json`。

## 科研决策：停止在 v6 上找最优阈值

当前证据足以排除的简单解释：**不是因为 Gate 总是选错，也不是因为 Token 调整几乎无法改变轨迹**。

仍不能区分的机制：

1. 任务难度和训练分布改变导致模型生成替代正确解的概率过低；
2. 固定512 tokens 对更多竞赛题输出存在不完整风险；
3. 任务提示格式（stdin / functional）、生成代码风格、执行器差异与成功率混杂；
4. SFT 微调可能使外部分布能力地板更加严重，但尚需同一批新任务上的 Base-vs-SFT 受控验证。

下一次实验必须在一个未参与本方法设计的**独立题目集合**上事先注册任务选择、模型/生成设置、正确性与数据泄漏保护。可考虑官方 LiveCodeBench 的独立 v5 题目作为较旧的分布级控制，但它不能用来宣称比 v6 更强的时间后推泛化或免污染证据；目前尚未开始任何新模型生成。

## 失败和工程留痕

- 初始 EXP-004M Wilson 区间测试误把 `0.9999999999999999` 与 `1` 做精确相等比较，导致 CI 失败；修正边界数值浮点舍入后真实 GH CPU CI 和分析 workflow 通过。
- 所有后验代码、异常、统计都通过新 commit 独立保存；没有修改 EXP-004K/L 的冻结配置和 `results/exp004l/summary.json`。
