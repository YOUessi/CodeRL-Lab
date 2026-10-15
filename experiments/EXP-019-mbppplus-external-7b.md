# EXP-019：7B SFT / 旧 GRPO / TRAIN-017 GRPO 的跨来源 MBPP+ 编程泛化评测

**首次登记：2026-10-15（模型生成或任何MBPP+模型评分前）。科研性质：历史公开基准的项目内新外部测试；不保证模型预训练没有见过MBPP(+)。**

## 为什么需要这一轮
TRAIN-017 的24道新算法留出题×2随机种子×每题4次×3个同源7B Adapter 共576程序，SFT/旧GRPO16步/新GRPO32步正确75/76/85，按题24簇 Bootstrap 新GRPO-SFT候选正确率+5.21pp（95%CI[+1.04,+9.90]pp），但四次覆盖31/48→32/48。已读EXP-018事后审计发现新增10条正确程序中9条来自早已覆盖的题目组合，主要是柱状图最大矩形问题的可靠性增加。**这是同训练题型、不同参数值的合成留出集。** 下一步应在来源和题型都不同的公开真实编程题测试相同3个7B权重，不能继续用该24题调参再宣称新盲测。

## 数据来源与严格选择冻结
- 标准基准：NeurIPS 2023 EvalPlus 的 MBPP+，源于经人工修订的 MBPP；固定 HuggingFace 数据仓库 `evalplus/mbppplus` 修订版 `8a14ef6256e2d9ba80a03e9ce57ee349a69e6399`，Parquet 相对路径 `data/test-00000-of-00001-d5781c9c51e02795.parquet`。Tang真实下载来源 SHA256 **`dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`**、378道题，字段task_id/prompt/code(参考实现)/test_list(公开示例)/test(隐藏的增广oracle)/test_imports。
- 为保证可用容器环境，先只筛 `test_imports==[]`、至少3条base示例断言、prompt非空且≤256字符、hidden augmented test非空（这些全部是基准元数据，**不得根据模型分数或哪题容易来挑选**）。从合格集合按固定SHA256 `sha256("EXP019|seed=20261015|"+str(task_id))` 升序选 **24道**。从已冻结的公共prompt中生成，不读取其 `code` 参考实现和增广 `test`。
- 模型输入提示为原MBPP+文字要求 + 官方公开的 `test_list` 例子，以指定函数签名/输出契约；**公开例子不是隐藏测试**，否则模型无从得知MBPP要求的函数名；只对隐藏 `test`（官方增广断言）在独立的评分阶段运行。题目、模型上下文和测试数据SHA一旦冻结，不允许更改提示或按测试分数过滤问题。
- 确保私有source/reference `code` 和增强oracle完全不进入public JSONL。原始数据存Tang，GitHub只存来源SHA、固定选择算法、输出摘要、Docker版本及每日过程记录。

## 统一模型/随机数/生成规则
- 同一个量化 Qwen2.5-7B-Instruct backbone revision `a09a35458c702b33eeacc393d103063234e8bc28`，NF4 double quant + BF16前向；同一个tokenizer/chat template，全部LoRA推理以FP32存储。
  1. **SFT（TRAIN-011）** `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`；
  2. **旧GRPO（TRAIN-016 16步）** `3d84428201f8802cdcde33cb64dd2c46ee4f2538de850cb2240b017ff2a237c8`；
  3. **新GRPO（TRAIN-017 32步）** `47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4`。
- 固定24题×两个种子 **42、314159**×每题**2次采样**×3个模型=**288条**候选，temperature0.8/top-p0.95/max_new_tokens256；三模型同题同种子同sample索引。必须明说本次是 `solved@2` 而不是 TRAIN-017 的 `solved@4`，不能跨不同采样预算直接比较覆盖率；如果未来要 `solved@4`，应事先另行冻结增量而非事后挑着扩展。
- 生成仅读取 `public_mbpp24.jsonl`，每条记录写入独立checkpoint及SHA。先固定生成24×2×2×3矩阵和汇总哈希，**然后才允许读取隐藏增广oracle**。禁止执行生成代码在宿主机。

## 官方测试环境约束与严谨性
- Docker 原生测试运行时使用已存在的 `semaremask-evalplus:latest`，固定镜像 ID `sha256:8734cd8b7868a1b0205384477cfdef64753ff979eb45d0249f60dc31a64c74ea`，镜像内支持 MBPP+ 增广断言所需要的 NumPy（已有EXP7C验证）。
- `--network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534`，单CPU、1GiB内存、限制进程数、候选级最长**30秒**，固定32/24道题依实际选择。模型代码与 `test` 都在隔离Python进程中的同一命名空间执行，捕捉AssertionError、TypeError、NameError、timeout等。**未知Docker基础设施/镜像问题必须报错停止，不得计为模型回答错误。**
- 在任何模型生成之前，可使用原始 MBPP+ 数据集中的reference `code` 对已选择24个oracle做独立Docker兼容性验收，并记录通过/超时/环境异常。但不得据此挑去模型解不出的题；如果oracle不能在锁定环境中执行，必须记录并创建显式新版本或修改全局统一评测环境，不能偷偷剔除后算分。
- 本指标应标注为：`MBPP+ 增广测试在本项目固定Docker容器下的验证通过率`；它不是直接复刻官方 EvalPlus CLI 的原样全套分组统计（包含预处理和官方不同的动态超时），不得伪称官方原榜单排名。
- 报告候选通过率/首样本正确数/`solved@2`、部分奖励（若无法获得标准化部分通过率则不报告）、任务级Bootstrap（24任务簇、两seed重复观测而不是48独立题）、错误类型及评测吞吐。保留没有完成的情况，不造可用性/质量结论。

## 预期交付与限制
建立本分支独立Docker测试器、源冻结器、24×2×2×3生成与评分入口、至少基础CPU测试、2题×三模型GPU smoke、全量完成日志。训练及评测不更改任何模型权重，也不执行新GRPO训练，避免在新外部holdout上调参。历史公开MBPP+可能与7B预训练数据重合，若发现正向差异只能说明该外部公开基准的描述性结果，不是新时间前瞻领域泛化。所有源码和汇总以GitHub为唯一事实源。
