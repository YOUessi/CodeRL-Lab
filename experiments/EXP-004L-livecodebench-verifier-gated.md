# EXP-004L：LiveCodeBench v6 Verifier-Gated Bottleneck Escape

## 状态

- 分支：exp/livecodebench-verifier-gated
- 基线：EXP-004K held-out 500-task 成功规则
- LiveCodeBench 官方 commit：28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
- 数据切片：code_generation_lite / v6（175 tasks）
- 数据接入：已完成，固定 source revision 和 SHA-256
- 12-task smoke：已完成（含官方私有测试验证）
- 正式外部分布：175-task GPU generation 已启动，尚未生成完整结果

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
