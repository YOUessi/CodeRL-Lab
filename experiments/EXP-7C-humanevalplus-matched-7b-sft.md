# EXP-7C：独立 HumanEval+ 上的同模型 7B 原始/SFT 配对能力边界对照

## 2026-10-14：在运行新任务模型生成之前冻结的协议

**目的：** 检验 CodeRL-Lab TRAIN-011 UltraChat SFT 对 Qwen2.5-7B-Instruct 在新代码题集上的可达正确程序集合有何影响。比较同一个基础模型的 **原始 7B Instruct** 和 **挂载已冻结的 TRAIN-011 LoRA Adapter 的 7B SFT**，不是与不同模型家族的 1.7B 再做不受控规模比较。

**背景限制：** 检查过 CodeRL-Lab GitHub 项目未发现 HumanEval+ 实验，官方 LiveCodeBench 仍只发布到 v6。选用 HumanEval+ 全部164题用于**相对于本项目此前 v4/v5/v6 的独立基准**；这是已公开多年的基准，不能宣称时间前瞻未污染或严格无训练泄漏，且 UltraChat 预训练数据可能间接包含相似代码。

### 冻结模型与来源
- 基础模型：`Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
- 7B SFT：`TRAIN-011` 4096条 UltraChat、231更新步冻结 LoRA Adapter SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`。
- 两组共用 NF4 double-quant、BF16 计算、相同 GPU、相同 tokenizer 和 native Instruct 模板，推理使用相同长度上限、采样温度、top-p、每道题随机种子及完全相同的**统一代码提取器**。
- 新测试源：`evalplus/humanevalplus` 数据集，**版本** `aa0d916268b1c17e84e881e9bd460508dd2fd308`，文件 `test.jsonl`，HuggingFace 给定 SHA256 `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`。运行时强制复核真实本地SHA和164个唯一任务ID，否则拒绝。
- 公开给模型的输入严格白名单：`task_id`、`prompt`、`entry_point`。参考答案、`canonical_solution`、`test`、`base_input`、`plus_input` 不进入提示或训练；首次评测前生成 checkpoint 具有不可变 SHA/完整度。
- 评测器：此版本冻结的 Hugging Face HumanEval+ JSONL 含原题代码提示、官方生成的增广断言测试字段 test；固定以整个 test 字段的 check(candidate) 为题级可执行正确性验收，**不把它再拆分为 HumanEval Base 与 Extra 两套分数**。Docker 隔离执行未经信任的候选程序，固定本机 EvalPlus 0.3.1 镜像 digest、12秒/候选、1GB内存、1 CPU、禁网络及无权限扩展。不得在宿主机直接执行模型生成代码；这不是 EvalPlus CLI 的分项 Base/Extra 原样评分。

### 生成与后处理的统一策略
- 模板：同一个系统指令「解决 Python 函数补全任务，输出完整可执行 Python 程序（含指定 `def` 与必要导入），不输出 Markdown 或解释」，用户消息包含完整原题 `prompt`，不加入答案或测试用例。
- 对两组各 **164题 × 4次采样 = 656个候选**，共1312条；种子是 SHA256(`EXP-7C|seed=42|task_id|sample_id`) 的确定性映射；两组每一题同一个 sample_id 使用相同种子。
- `temperature=0.8`、`top_p=0.95`、`max_new_tokens=384`、`do_sample=True`。除非显存失败，严格不按模型表现改动；任何改变以新的协议版本另立实验。
- 代码提取固定：若开头为完整的 ```python` 代码围栏，选择**第一个**完整代码块去围栏；若缺失闭合围栏则保留原始文本；其余原始文本原样保留。**不改算法、不凭题目/正确率筛选、不自动补导入或修逻辑**。生成身份、样本原文和转换后代码哈希全部保存在 Tang 本地；GitHub只上传汇总、哈希、脚本。
- 工程 smoke 从固定排序的任务前2题、两臂各1次采样验证 GPU 加载/Adapter 切换/解码/ checkpoint/ Docker；这些不计正式研究证据，正式164×4另起冻结目录且不能把smoke数据混入。

### 研究指标与边界
- 每组**固定 HumanEval+ test 增广断言**的 solved@4（164题至少一个正确程序）、候选级正确率，样本级 Python 语法/函数入口错误率，固定超时、资源限制和失败类型。此冻结源只提供一个组合断言 test 字段，不能声称分别复现官方 HumanEval Base/Extra 两个独立指标。
- 同题同采样ID配对统计：SFT与基础模型在`correct`/ `wrong`的迁移四象限；20,000次按任务重采样 Bootstrap (seed42)，报告差异的 95% CI，不能只是看单个百分比。
- 重要审计：Python语法可解析率、Markdown格式修复比例、超长截断比例、任务函数名称契约、非代码解释率、容器资源限制/超时/环境故障数、GPU显存峰值。
- 发现错误/中断/负面结果，必须按日期写入 GitHub `docs/daily/`，包括复现、根因与修复。不能把“模型输出变了”当作“正确程序覆盖增加”。

### 与两条线的关系
- **A线**：这是真正 TRAIN-011 已训练 SFT Adapter 与原模型在完全一致代码输入上的独立迁移验收，不只验证 UltraChat NLL。
- **B线**：进一步检查 SFT 会不会改变多次采样的答案覆盖，以及是否有 SFT wrong→correct 或 correct→wrong。
- 不在本阶段引入/调整局部低 Margin 门控。只有判断清楚 SFT 对可达解分布的作用后，再考虑一条独立、提前冻结参数的干预对照。

参考：EvalPlus 官方文档 https://github.com/evalplus/evalplus/blob/master/docs/cli.md ，HumanEval+ dataset https://huggingface.co/datasets/evalplus/humanevalplus/tree/aa0d916268b1c17e84e881e9bd460508dd2fd308
