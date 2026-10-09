# EXP-004N：在独立题集上区分解题能力地板与长度限制

## 研究状态（2026-10-08）

- 阶段：**独立研究问题预注册 + 数据入口准备**。
- GPU 阶段尚未执行新的 v5 生成、私有测试或成效计算；本分支已经实现独立的六臂生成与官方 private scorer，正进行前置工程验证。EXP-004L 的175题不用于调参。
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

## 2026-10-08 15:45 后：EXP-004N 六臂可执行流水线已实现（未产生结果）

### 主要干预与模型能力对照完整冻结

- 冻结的 SFT 512-token policy：`src/coderl_lab/analysis/livecodebench_verifier_gated.py --experiment-id EXP-004N`，保持 512/128/0.05/0.20/0.25 以及 public-fail Gate，不改变 EXP-004L 默认逻辑。
- Base512 原始模型与 SFT1024 同一模型家族控制：`src/coderl_lab/analysis/exp004n_capacity_generate.py`；仅使用官方 GenericBase prompt，不读取 private tests。
- 正式六组条件：Base512、SFT512、gated-low、gated-high、always-low、SFT1024。
- 官方私有测试 scorer：`src/coderl_lab/analysis/exp004n_formal_eval.py`；先检查 v5 完整167题、public view SHA、模型/Adapter SHA、官方 checkout commit、三份 frozen runner 哈希，再开始第一条 private 解码；checkpoint 仅记录布尔正确性与公开错误类别，不存 private I/O。
- 主要结果仍仅使用 gated-low - SFT512 的 paired task bootstrap 20,000 / seed42 / lower CI > 0，次要比较分别为 Base512-SFT512 与 SFT1024-SFT512。
- 运行入口：`scripts/run_exp004n_generate.sh`（GPU，三份 runner 冻结）与 `scripts/eval_exp004n.sh`（冻结后正式私有测试）。
- 此阶段**没有生成正确率结论**，尤其不能把 v6 的14/175结果迁移为 v5结果。

### 测试失败记录与修复

- 新增 CPU 单测模拟167任务、私有数据未触及、公共门控/输出 Token cap/AST 指标以及配对 Bootstrap。
- 最初的 synthetic task-grid fixture 缺少 `task_id`，触发 `bootstrap_task_cluster` 对该字段的严格要求，真实 GitHub CI 失败；通过在 scorer 的真实任务详情和 synthetic fixture 中都写入 `task_id` 后修复。
- 最新 GitHub CI 在修复后通过：对应提交 `31a99cd8a33797ffd002a89dba343c16ac2989a2`，是数据/统计输入契约问题，而不是模型科研结果。

### 实际数据接入与硬件互斥

- EXP-004N 的 v5 public-only 167题、v6 ID/题面 disjointness 与 source/public SHA 早已由 [独立 GitHub Actions](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37729627752) 真正验证。
- 15:48 已在 Tang 的 B 工作树拉取分支并启动 `python -m coderl_lab.datasets.livecodebench_v5`，只下载 fixed source 和产生 public-only view；网络读取仍在继续，无 private 解码。
- 同时 A 分支正式 4096条 BF16 LoRA GPU 已在训练，故 B 不启动新的 CUDA 生成；通过独立 CPU/网络路径继续研发，保证不抢单 GPU。


## 2026-10-08 15:58：v5/v6 可观察任务分布差异（模型结果盲）

先固定并核验 v5 public SHA `e695ba9fa2ce35abc8db2f3360bf711930746cd55843890177ecd518c0a4c98d` 和 v6 public SHA `f9fd88d4e1b35b4f6720ca548c2e2d1187ad53b5fb5723ef4999a40b79d7c399`。随后使用只含公开题面/元数据的程序 `exp004n_public_shift.py` 真实对比全部题，且所有指标**不依赖模型输出或 private tests**：

| 类别 | v5 (167) | v6 (175) |
| --- | ---: | ---: |
| Easy | 41 | 43 |
| Medium | 52 | 52 |
| Hard | 74 | 80 |
| AtCoder | 105 | 112 |
| LeetCode | 62 | 63 |
| Public tests | 441 | 463 |
| 平均题面字符数 | 1476.22 | 1404.81 |

Hard 比例 v6 比 v5 仅高约1.40个百分点。其余 difficulty/platform 构成也接近。

**有效结论：**难度/平台的粗粒度配比没有发生剧烈改变，不支持仅用“v6 的 Hard 比例大幅升高”解释迁移失效。但这个表格不能证明算法难度相同，也不能预测 Base/SFT 的正确率。实际能力地板、答案完整性和推理门控仍需冻结生成及 hidden 测试才能检验。

机器可读：`results/exp004n-pretest-shift/summary.json`。原始任务未被选择性排除，也没有因为此诊断调整任何生成条件。

## GPU 独占的正式运行入口

- `scripts/run_exp004n_generate.sh` 固定全部167题，先SFT512/gated-low/high/always-low，再Base512及SFT1024；任何已存在的 frozen runner 会阻止覆盖。
- `scripts/eval_exp004n.sh` 对全部三份 frozen runner 完成 SHA、模型/Adapter、官方版本和 gate 复核后，才读取 `test5.jsonl` 私有测试并执行官方 Docker。后验正确率、paired Bootstrap 20,000/seed42 和 Gate/资源错误审计只在这一阶段出现。
- `scripts/queue_exp004n_after_train.sh` 是可选的 Tang 本机 GPU 独占串行启动器：必须验证 A 的完整 BF16 LoRA Adapter 确实保存、B 源代码 SHA 未改变以及无外部 GPU compute 进程。它不是统计结论，也不是 ChatGPT 异步通知服务。
