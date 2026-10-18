# 第一阶段复现与验收入口

[首页](../../README.md) · [总报告](../reports/STAGE1_REPORT_ZH.md) · [版本与演示](../releases/STAGE1.md)

## 0. 先明确复现的三个层级

| 层级 | 实际执行 | 所需资产 | 能证明什么 |
|---|---|---|---|
| L1 公开结果复算 | 重算96题聚合计数、主比较和可选区间 | 本仓库、Python；区间需 NumPy | 发布表格与公开数据一致，不证明重新训练成功 |
| L2 所有者产物核验 | 只读检查原始分数、生成、采样键、模型和奖励日志哈希 | Tang 原始 artifacts | 公开计数与原始产物对应；不重新运行候选代码 |
| L3 重新执行 GPU 实验 | 重训、推理、容器判题 | 模型/SFT/任务/隐藏测试/镜像/专用环境 | 新的一次实验复现；本交付没有执行这一层 |

稳定快照不附带 L3 所需的大型或受访问控制资产。不得把 L1 或 L2 的成功写成“全部 GPU 实验已经从零重跑”。

## 1. 获取正确版本

固定研究快照标识为 `stage1-7b-v1.0.0`；准备分支为 `release/stage1-7b-20261018`。最终是否已经打标签以及验收提交，以[版本验收记录](../releases/STAGE1_ACCEPTANCE.md)为准；在标签可见之前使用分支，不猜测版本已发布。

从仓库根目录运行下述命令。默认入口只使用标准库：Python 3.10+，无网络、无容器、无模型加载、无文件写入。它不会修改已有训练产物。

```bash
python scripts/stage1_release.py demo
python scripts/stage1_release.py demo --json
python scripts/stage1_release.py verify
```

预期关键结果：SFT 202、原有题型43 202、宽覆盖43 201、原有题型44 203、宽覆盖44 207；每个模型的分母384。输出状态为 `RECOMPUTED_COUNTS_NOT_NEW_EVALUATION`。

展示过程中保持三个指标分开：正确候选/384、固定 sample0 正确/96、四次至少一次解出/96。公开 CSV 只有每题计数，不含测试输入、正确答案或生成程序。它足以复算主指标与任务级区间，但不是完整候选级翻转明细。

## 2. 验证统计区间与测试

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/stage1_release.py verify --bootstrap
python -m pytest -q
```

`.[dev]` 由本快照的 pyproject.toml 声明，包含 pytest 和 NumPy。`verify --bootstrap` 对相同96题按任务聚类重采样20,000次，随机种子20261018，复算三个主指标区间。它与历史报告保持相同算法，不把384份候选当成384道独立题。

重要预期值，均为“宽覆盖−原有题型”的候选成功率差：

- seed43：−1/384；区间 [−0.0130208333, +0.0078125]。
- seed44：+4/384；区间 [−0.0026041667, +0.0260416667]。

这里是比例，不是百分点；展示为百分点需乘100。两种子的区间都包含零，不应写成已确认正收益。

## 3. 在 Tang 只读核验原始证据

仅当拥有原始产物时运行；默认根目录不写死，必须显式传入。实际所有者布局为：

```text
/home/you/projects/CodeRL-Lab-track-train017/artifacts/
  exp028_v3/
    four_trainings.json
    data/{narrow,broad}/train.jsonl
    {narrow,broad}_seed{43,44}/
      summary.json
      adapter_model.safetensors
      reward_attempts.jsonl
    evaluation/
      generation_freeze.json
      generation_<adapter-sha-prefix>/checkpoint.jsonl
  exp029_eval4g/
    final.json
    reference_results.jsonl
    scored_<adapter-sha-prefix>.jsonl
```

在稳定版本的源码目录执行：

```bash
python scripts/stage1_release.py audit \
  --artifact-root /home/you/projects/CodeRL-Lab-track-train017/artifacts
```

核验内容包括原 final.json 固定哈希、四个训练模型与128条奖励记录、五模型的1,920个采样位置、生成正文SHA、采样种子、原始判题文件SHA、96个参考解通过记录，以及公开聚合表是否逐题吻合。生成正文仅用于重新计算哈希，不执行、不打印。此命令不读取外部隐藏测试正文，也不调用 Docker、GPU 或网络。

预期状态为 `OWNER_ARTIFACTS_VERIFIED_READ_ONLY`。资产缺失、哈希不符、样本重复或不足必须非零退出，不自动下载替代、不覆写原文件、不把缺失数据解释为零成绩。

这不是对 Docker 运行事实的独立外部公证；它建立的是版本化源码、原始记录与公开表格之间的可核验对应关系。签名、第三方重跑和预训练污染保证不在本次交付范围内。

## 4. 完整 GPU 复现需要什么

本次没有启动新的 GPU 训练。历史训练和评分实现保留，以便以后明确授权的新复现使用，但并未包装为“陌生机器上下载即重训”的通用工具。

| 资产 | 标识与获得条件 |
|---|---|
| 基础模型 | `Qwen/Qwen2.5-7B-Instruct`，revision见发布清单和配置；需要本地缓存及上游允许的访问 |
| 冻结 SFT 适配器 | SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`；本快照不附权重 |
| 训练数据及生成器 | 源码在 `src/coderl_lab/datasets/`；具体已冻结JSONL与manifest保留Tang；再生成后也必须逐文件核对 |
| 外部评测源 | 固定 MBPP+ Parquet及选择manifest；不能直接取最新 main 替代已固定SHA |
| 隐藏测试 | 仅在模型答案完整冻结后可由评分进程读取，不进入训练或提示 |
| 环境 | 历史记录为 Torch2.10.0+cu128、Transformers5.18.0、TRL1.14.1、PEFT0.21.2；实际环境快照见下方说明 |
| 量化与推理依赖 | 需要匹配 CUDA、bitsandbytes、accelerate、safetensors 等；现有 pyproject 的 model extra 不是全环境锁文件 |
| 容器镜像 | 训练与外测为不同固定镜像ID；不能只按可变 tag 认定等价；镜像二进制不随仓库发布 |

最终覆盖训练入口为 [run_exp028_resource_v3.py](../../scripts/run_exp028_resource_v3.py)，最终修订评分入口为 [exp029_frozen_eval.py](../../src/coderl_lab/analysis/exp029_frozen_eval.py)。这些是保留了 Tang 主机名、原路径和不可覆盖约束的研究脚本，不是跨平台部署 CLI。

可以先只检查参数说明。显式设置 `PYTHONPATH=src`，避免共享环境中的旧 editable 安装解析到其他工作树：

```bash
# 需要已安装的研究依赖；--help 不代表复现已经完成
PYTHONPATH=src python scripts/run_exp028_resource_v3.py --help
PYTHONPATH=src python -m coderl_lab.analysis.exp029_frozen_eval --help
```

不要为了重跑而删除 `identity.json`、`failure.json`、`chain_failure.json` 或模型目录。新复现需先固定独立输出目录与资源协议；原入口对历史目录的约束需要明确改造和记录，不能靠手工替换哈希或放宽校验绕过。原任务不因本次文档交付自动重启。

[当前 Tang 依赖快照](../../releases/stage1/tang_environment.json)记录本次交付核验时的已安装版本，不等于对每个历史实验的环境宣称；历史实验以各自 summary.json 中的版本字段为准。

## 5. 固定资产与来源

[manifest.json](../../releases/stage1/manifest.json)记录公开计数哈希、上游研究提交、最终实验摘要哈希、原模型哈希、生成冻结哈希与原评分文件哈希。`demo` 和 `verify` 都检查这些公开计数的内部对应；`audit` 再与 Tang 原始文件对应。

实验源码来自研究提交 `a745222bda135910af0f62f20574bc98a45a5ffc`。本版本增加的是文档、计数和只读入口，不修改研究中的奖励、模型、任务、采样或预算。最终交付提交由验收记录给出，不能把研究提交与发布提交混为一个标识。

## 6. 常见问题

**只有旧 README？** 检查是否仍在 main 或旧实验分支。本轮稳定版本有独立入口；main 不被自动合并。

**没有 NumPy？** `demo` 和普通 `verify` 仍可运行；只有区间复算与项目 dev 测试需要依赖。

**缺少 Tang 的 artifacts？** 只能做公开计数复算，不能声称所有者产物核验通过。程序应明确退出，不自动造数据。

**看到历史 Run failed 或停止记录？** 这些记录不删除；当前发布CI与过去运行分开看。参考解95/96不是模型正确率，4GiB新协议也不是原1GiB协议的成功复现。

**本次暂停训练是否意味项目失败？** 不是。本版本交付的是在明确条件下经过验证的研究过程及有限结论，而非必须上涨的指标。
