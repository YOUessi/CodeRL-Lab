# EXP-004N：在独立题集上区分解题能力地板与长度限制

## 研究状态（2026-10-08）

- 阶段：**独立研究问题预注册 + 数据入口准备**。
- 当前尚未执行任何新的 v5 模型生成、私有测试或成效计算；EXP-004L 的175题不用于调参。
- 所有正负结果必须保留。

## 为什么不是继续修改 EXP-004L？

此前正式175题：

- Baseline 14/175，gated-low 15/175，95% CI 下界=0；
- Gate 可在148题中识别真实错误，但低 margin 改变119条轨迹，仅救活1题；
- 41/175 原始输出达到512生成 token上限，28/175 原始代码AST无法解析，28题全部触及上限；
- 不能仅靠在已见 v6 上修改输出长度、模型或阈值后宣布原方法跨分布成功。

因此**在任何新候选生成前**固定下一组研究假设、数据、模型与结果协议。

## 独立数据源

- Official repo：`livecodebench/code_generation_lite`。
- Pinned dataset commit：`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`。
- 新题集合：**release v5 only**（官方 `test5.jsonl`，不是 v1-v5 的并集）。
- 使用**全部 v5 题**作为新验证队列，实际题目数从独立 public-only manifest 确认；不能在看到正确率后挑选 difficulty/平台子集。
- 每题先只导出 question / starter code / public tests / metadata / date / difficulty；不解码、不导出 private tests。
- 数据准备阶段只检查任务数、唯一 question_id、是否与 v6 的已使用题目有交叉，执行器格式以及 SHA-256。若重复 question_id 则应事先剔除重复项并记录；不能根据正确率筛选。
- v5 是比 v6 更早的题目时间窗口，因此只能作为**之前未用于此方法选择的题集级分布对照**，不能称为更前向的2025以后无污染评测。

## 预注册假设与判据

### 主要假设 H1：冻结策略迁移

保持 EXP-004J/K/L 的 SFT policy 和 decoding：

- 模型 Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1
- 原 EXP-006A SFT adapter（验证 SHA）
- greedy；max_new_tokens=512
- window128；low<=0.05；high>=0.20；bias=-0.25
- Gate=baseline public_fail AND eligible
- baseline / gated-low / gated-high / always-low 对照
- 按题配对 bootstrap 20,000 次，seed42
- 成功标准：gated-low - SFT baseline 的95% CI **下界严格>0**。

失败则明确保留负结果。不得在看见 v5 隐藏测试结果后调阈值或删除 Hard 题。

### 次要假设 H2：模型能力地板

用**同一批未见 v5 题与官方输入模板**评测 Qwen3-1.7B 原始 Base greedy512（无需 LoRA）。比较其和 SFT baseline 的总正确率、平台/难度分层、公共测试命中情况。

不能把这一步的 Base/SFT 差异归因于单个因素，因为模型训练与预训练污染不可控。

### 次要假设 H3：512 token 截断是否影响有效程序

在同一 v5 题上，对原始 SFT greedy 新增 **1024 token** 上限条件，与512 token成对比较，保持其余模型参数、输入模板一致；聚合 hit-cap、AST parseability 和官方 hidden correctness。

**1024属于预注册的次要条件，不回填替换 H1 的512-token结果**。不能在 v5 上决定新的“最佳窗口”或 -0.25 bias 后再次把同一批题当作独立测试。

### 其它必须报告的控制

- gate selected hidden-wrong precision/recall；
- intervention applied vs changed fraction；
- wrong→correct 和 correct→wrong；
- public-pass 保护；
- 公开/私有测试的执行错误类别、超时和内存；
- 代码提取策略固定，不能仅对某一实验分支手工修复输出。

## 当前真实工程实现范围

- 完成 `src/coderl_lab/datasets/livecodebench.py` 的 stream public-only v5/v6 参数化，保留 v6 默认行为；
- 增加 `tests/test_livecodebench_adapter.py` 的 v5 私有数据不外泄、v6默认不变与 question_id 唯一性测试；
- 提供 GitHub Actions CPU-only 的 v5 固定数据预处理；完整模型/推理对照暂未完成，也**不冒充已完成**。
- GPU 应优先分配给双主线 A 的 TRAIN-007A 正式4096样本训练；B 的新 GPU generation 必须在 A 结束后再运行。

## 不可能从当前实验直接得出的结论

- “大模型能力差是 v6 负结果的唯一原因”；
- “调高 tokens 就能恢复 token escape”；
- “v5 会复制 MBPP +1.60 pp”；
- “任何分布的 public Gate 都无用”。


## 2026-10-08：独立公开数据集冻结完成

- GitHub Actions [EXP-004N 数据构建及跨 release 去重检查](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37729627752) 已成功。
- 官方固定版本 `test5.jsonl` 仅包含 v5 增量 **167题**，并与先前使用的 v6 175题进行 question_id 与规范化题面+starter_code 双重交叉核验；两类交集均为 **0**，因此没有排除任务，也没有按难度筛选题目。
- v5 raw SHA256 `7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214`；公开视图 SHA256 `e695ba9fa2ce35abc8db2f3360bf711930746cd55843890177ecd518c0a4c98d`。
- 对照 v6 的原先固定 public-view SHA256 也经同一数据源 commit 重建验证为 `f9fd88d4e1b35b4f6720ca548c2e2d1187ad53b5fb5723ef4999a40b79d7c399`。
- 机器可读记录 `results/exp004n-data/summary.json`，公开任务与 manifest 保存于 [GitHub Actions Artifact](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37729627752/artifacts/11529138203)，原始私有测试从未解码或导出。
- 本条记录仅是 GPU 生成前的数据冻结；主检验及 Base/SFT/1024 控制没有执行，不能发表新方法效果结论。
