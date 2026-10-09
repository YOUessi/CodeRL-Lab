# EXP-004N：外部早期时间窗口的 Base-vs-SFT 能力地板对照

**状态：2026-10-08 实验设计与数据准备；尚未运行 GPU。**

## 动机

EXP-004L 在 LiveCodeBench v6 175题上，SFT greedy 仅14/175、Hard 0/80；有148题触发 public gate，119题改变轨迹，但只修复一题。后验 EXP-004M 提示：问题不是简单“干预没有生效”，而是改后的正确候选稀少。不能仅根据 v6 已知结果调门槛、bias 或 max token 后冒充独立正向复制。

## 时间窗口与研究限制

选择 LiveCodeBench `code_generation_lite/test5.jsonl`，即官方细分 `v5` 新增题，而不是整个 `release_v5`（后者包括更早版本）。使用与 EXP-004L **相同上游数据仓库 revision** `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`；官方 Python evaluator 仍固定 `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`。v5 是比 v6 **更早**的任务，不能宣称比 v6 更强的前瞻泛化证据；此实验是探索性时段/能力地板对照。

## 预先规定的方案

1. 公共数据准备：只把 source 的题面、starter_code、公开测试、来源元数据导出，不解析 private tests。
2. 先构建 task_id 去重、public-test 分类、难度与年份统计、source SHA256、public view SHA256；以 manifest 固定任务数后再运行任何模型。
3. 第一研究比较：Qwen3-1.7B **Base vs EXP-006A SFT adapter**，同官方 GenericBase one-shot prompt，greedy 512-token，不进行 Token 干预。
4. 评估两臂在完整独立 v5 子集上的 private exact correctness；先生成并冻结两臂原始完成，后使用同一个官方容器与冻结私有测试评估。
5. 成对统计 Base-vs-SFT 正确率、wrong→correct、correct→wrong、任务级 Bootstrap 20,000 seed42；同时报告平台、难度、token cap、AST 解析性，以及完成长度。
6. 使用结果不能回头更改 EXP-004L 正式结果；任何 Gate 干预迁移必须另立预注册实验，不能从本项目挑 v5 有利题调参后宣称复制成功。

## 假设与竞争解释

- H1 能力地板：即使使用 Base，其通过率在 hard 题仍接近零，说明 v6 的低 correct coverage 不主要来自该 SFT adapter 的特异性退化。
- H2 微调分布偏移：Base 显著优于 SFT，提示 MBPP SFT 可能对竞赛任务造成迁移退化，但不能仅凭 observational pairing 判定训练数据污染或某个具体中间机制。
- H3 截断混杂：长度上限与 AST 无法解析集中在 hard 题；仍需要独立检验功能错误，不把 AST parse 当作 hidden correctness。

**重要：**本实验不可用 v6 的已知 hidden outcome 为 v5 选择任务，不可在 v5 上调整阈值并返回污染 EXP-004L；v5 是诊断性评测，不是超参数开发集。

## 执行入口

- `python -m coderl_lab.datasets.livecodebench_v5 --output-dir artifacts/exp004n/data`
- `results/exp004m/` 保存此前 v6 的后验诊断；不得混写 v5 指标。
- 只有完成验证脚本和数据冻结后，才开展模型生成、官方 private tests 与 Bootstrap。
