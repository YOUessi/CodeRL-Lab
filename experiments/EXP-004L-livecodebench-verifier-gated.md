# EXP-004L：LiveCodeBench v6 Verifier-Gated Bottleneck Escape

## 状态

- 分支：exp/livecodebench-verifier-gated
- 基线：EXP-004K held-out 500-task 成功规则
- LiveCodeBench 官方 commit：28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
- 数据切片：code_generation_lite / v6（175 tasks）
- 数据接入：已完成，固定 source revision 和 SHA-256
- 12-task smoke：已完成（含官方私有测试验证）
- 正式外部分布：175题官方 private evaluation 完成，primary 不显著（+0.57 pp，CI 下界=0）

## 研究目的

EXP-004K 已经在未参与方法设计的 MBPP test 500 tasks 上复制成功：

- baseline 41.40%
- gated-low 43.00%
- +1.60 pp
- 95% CI [+0.40,+3.00]

现在不再在 MBPP 上调整任何 threshold / bias / window。

本实验检查：

> public verifier + local low-margin bottleneck escape 是否能跨到时间更新、竞赛式、stdin/functional 混合代码任务。

## 冻结规则

全部沿用 EXP-004K：

- model：Qwen3-1.7B-Base + EXP-006A SFT adapter
- greedy decoding
- max_new_tokens=512
- primary window=128
- low threshold=0.05
- high threshold=0.20
- bias=-0.25
- gate：baseline public tests 未全部通过 AND low bottleneck exists
- gated-low / gated-high / always-low 定义保持不变
- bootstrap 20,000
- seed=42

## 数据

官方数据：

- repository：livecodebench/code_generation_lite
- fine-grained version：v6
- raw file：test6.jsonl
- tasks：175
- 时间窗口：release_v6 新增题，约 2025 年新题
- 字段：
  - question_content
  - starter_code
  - public_test_cases
  - private_test_cases
  - metadata
  - contest_date / platform / difficulty

选择 v6 而不是 release_v6 全量 1055 题，是为了：

1. 更强分布外性；
2. 降低可能的早期 benchmark contamination；
3. 控制 1.7B 单卡实验成本；
4. 使用官方支持的 fine-grained release slice。

## 无泄漏协议

### Phase 0：public-only 数据准备

只读取：

- question / starter code
- public_test_cases
- metadata.func_name
- release metadata

不解码、不导出 private_test_cases。

### Phase 1：SFT baseline + margin trace

GPU 只做：

- greedy baseline
- first128 margin profile
- low/high bottleneck position

### Phase 2：public verifier

只执行 public_test_cases。

得到：

public_fail AND low bottleneck exists

### Phase 3：second-pass

同一个 SFT adapter：

- gated-low：low bottleneck -0.25
- gated-high：high-margin control -0.25
- always-low：所有 eligible task 都 low -0.25
- baseline：不改

### Phase 4：官方 final evaluator

只有 Phase 1-3 完全冻结后：

- 调用官方 LiveCodeBench evaluator；
- 读取 private_test_cases；
- 评估 baseline / gated-low / gated-high / always-low。

## 官方 evaluator

不自己复写 LiveCodeBench checker。

固定官方仓库 commit：

28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24

CodeRL-Lab 只导出：

- question_id
- code_list

最终执行使用官方 lcb_runner code-generation evaluator。

## 预注册主要结果

Primary：

gated-low hidden correctness - baseline hidden correctness

成功标准：

95% paired task bootstrap CI lower bound > 0。

Secondary：

- gated-low - gated-high
- gated-low - always-low
- wrong→correct / correct→wrong
- public-pass task protection
- gate precision / recall

## 当前已遇到的数据接入问题

### 问题1：datasets 5.1 不再支持 dataset script

老式：

load_dataset(... trust_remote_code=True)

在当前环境失败：

Dataset scripts are no longer supported.

决定：

不降级 CodeRL-Lab 的 datasets 版本。

### 问题2：Parquet 自动转换文件不在 main

尝试 main/release_v6/*.parquet 返回 404。

原因：

Hugging Face Parquet 自动转换位于单独 revision，不在 dataset main。

处理：

正式准备脚本改为直接下载 upstream main 的 test6.jsonl，并自行解析 JSONL；
不依赖远程 dataset script，也不依赖 Parquet 自动转换 revision。

## 工程原则

- GitHub 是代码/配置/日志唯一事实源；
- Tang 只下载数据和跑真实 GPU / official evaluator；
- private tests 不写入 Git；
- public-only manifest 可以记录 hash/count，但不包含 private content；
- 所有失败、修复、数据 hash、官方 commit 写入每日研发日志。


## 2026-10-08：分布外 smoke 实测

固定数据源：

- revision：`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`；
- raw `test6.jsonl` SHA-256：`bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`；
- public-only view SHA-256：`f9fd88d4e1b35b4f6720ca548c2e2d1187ad53b5fb5723ef4999a40b79d7c399`；
- 175 unique tasks：AtCoder 112 / LeetCode 63；easy 43 / medium 52 / hard 80；public test cases 463；
- contest dates：2025-01-04 至 2025-04-06；private 数据未导出、未在选题/门控中读取。

Smoke 采用结果盲、预先固定的 2平台×3难度×各2题，确保测试 stdin/functional 两类执行，而非只用默认排序后的前12题。

- 完整 CPU 回归：149/149 通过；
- GPU 12/12 完成；eligible 9/12；gate triggered 9/12；public-pass 1/12；
- runner private_tests_accessed=false；
- official final scorer 严格在 runner 冻结后首次解码 private tests，成功完成；
- Baseline / gated-low / gated-high / always-low 全为 1/12；三项与 Baseline 的净差均为0；
- wrong→correct=0，correct→wrong=0；
- smoke 正确率不作为 efficacy 证据，不据此改变 threshold、bias、window 或 prompt。

机器可读结果：`results/exp004l-smoke/summary.json`。

### 正式175题

已在 Tang 启动正式生成，固定同一代码、模型、数据与解码规则。必须满足 175/175、gate audit 和 runner 完全冻结之后，才能进入 official private evaluation；独立统计器执行 paired task bootstrap 20,000 / seed42 和三种对照。尚无175题正确率可报告。


## 2026-10-08：175题完成生成，官方判分 137 资源限制修复

### 正式生成冻结

- 175/175 生成完成；eligible=159；actual gate triggered=148；baseline public-pass=17；
- `runner.json` 记录 `private_tests_accessed=false`；
- 已复核 Gate 语义与 public-pass 保护，之后才启动 private official evaluator；
- 固定 model、greedy、window128、low0.05、high0.20、bias=-0.25、memory1g、timeout6s，参数没有因结果而改变。

### 正式评测中断，不得冒充负样本

- private official evaluator 在第25题之后的 `abc391_f/gated_low_destabilize` 遇到原先被归为 `runner-error` 的 Docker 进程退出 `137`；
- 原因调查：重跑**完全相同的冻结代码、任务和1g Docker配置**，baseline可以由官方 checker 返回普通的错误判定，gated-low 则在约2.7s 发生无 stdout 的 exit 137；
- exit 137 = SIGKILL，结合固定 `--memory 1g` 资源限制，这是候选代码特定的**资源限制违例**，不能把它作为官方评测服务整体失效，更不能因此增加 hidden evaluation 的内存上限；
- 修复仅作用于评测器错误分类：将原先 exit137 的笼统 `runner-error` 显式记录为 `candidate-resource-limit`，作为错误代码（passed=false）统计，其他无输出的容器错误继续保持基础设施故障并中断；
- 冻结 runner hash、已评测缓存、对照参数、数据、私有测试及官方 checker commit 都不变；
- 扩充 CPU 单元测试：exit137 → candidate resource-limit，exit125 → 真实 Docker 启动错误；增强 checkpoint 记录与每个 arm 的资源限制计数；
- 修复提交：`2a382f5cb2caff267d76f8c1a0367b7c8a56ad4d` / `46cf6d4db97a2e26402b14773481efbda320bf64`。

### 下一步

保持已有 25题后续 checkpoint，不删除已经完成的正式私有测试结果。从原始 `runner.json`（175题已冻结）恢复官方 evaluator，完成 175题后再统计主要配对 Bootstrap 和对照；**尚未得出175题正式准确率**。

## 2026-10-08：LiveCodeBench v6 175-task 正式结果

### 数据、方法与边界

- 175 / 175 题全部完成冻结生成与官方私有测试（AtCoder 112，LeetCode 63）；
- no-private-before-freeze=true，固定官方 commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`、数据源版本和输入 SHA；
- 175题里 eligible 159、actual public-fail + low-margin Gate 触发148、baseline public-pass17；
- 全程固定 greedy、max tokens512、window128、low0.05、high0.20、bias-0.25、Docker memory1g / timeout6s；
- paired task bootstrap 20,000，seed42；
- Docker returncode 137 经同任务同资源复现，被显式归为 candidate resource-limit，不改变内存上限与模型输出；各 arm 均1例。

### 正式正确率

| Arm | Correct / 175 | Accuracy | Delta vs SFT |
| --- | ---: | ---: | ---: |
| Baseline SFT greedy | 14 | 8.00% | — |
| Gated-low | 15 | 8.57% | +0.57pp |
| Gated-high control | 14 | 8.00% | 0 |
| Always-low without gate | 15 | 8.57% | +0.57pp |

Primary (gated-low - baseline)：+1 / 175 = +0.5714pp，95% paired-bootstrap CI = **[0, +1.7143] pp**，20,000 iterations，seed42；预注册标准 lower bound>0 **未通过**。

错误转换：wrong→correct=1、correct→wrong=0；gated-high 完全无变化；always-low 取得与 gated-low 相同的净收益，因此**外部分布没有复制出 Gate 必要性优势**。

按平台/难度（仅描述，非预注册主要结果）：

- AtCoder：12/112 → 13/112；LeetCode：2/63 → 2/63；
- Easy：11/43 → 12/43；Medium：3/52 → 3/52；Hard：0/80 → 0/80。

后验 gate audit：被选中148题均为 baseline hidden wrong，precision=100%，recall=148/161=91.93%；当 baseline 在 175题上仅8%正确时，不能把 gate precision 很高误解为 decoding intervention 一定有效。

### 研究结论

EXP-004J/K 的 MBPP 500题独立同家族复制成功，不等于跨任务分布有效。本次 LCB v6：

1. 点估计 +0.57pp，但只有一题 rescue，95% CI 包含0，不满足预注册成功标准。
2. high-margin control 仍未见收益，但 ungated always-low 也达到同一提升，没有复现 K 中 Gate 必要性证据。
3. 当前1.7B Base+MBPP SFT 在竞赛式任务上基线仅8%，Hard为0/80；这提示明显的任务难度/能力地板效应，但**未经额外对照不能断言这是唯一原因**。
4. 不因负结果调整这175题上的阈值、prompt、bias 或 max_new_tokens；在后续新模型/任务实验重新预注册前，先保持此负结果和全部数据版本。

机器可读正式结果：[results/exp004l/summary.json](../results/exp004l/summary.json)；分支 PR #27 保持 Draft，待科研结论审阅。
