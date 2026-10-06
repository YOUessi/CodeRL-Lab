# 正式数据集方案 v1

## 目标

CodeRL-Lab 的数据层必须同时满足四个需求：

1. 有可直接执行的监督信号，可用于监督微调；
2. 有公共测试，可构造可验证训练奖励；
3. 有与训练隔离的隐藏测试；
4. 有独立外部基准，避免只在一个老数据集上得出科研结论。

因此不采用“一个数据集包打天下”，而采用三层结构。

## 第一层：MBPP —— 训练与同分布基线

数据源：`google-research-datasets/mbpp`。

原始 MBPP 约 1000 道基础 Python 编程题，每题提供任务描述、参考代码和自动测试。

采用官方常见 ID 分割：

- train：374 题；
- validation：90 题；
- test：500 题；
- prompt/few-shot：10 题，不进入训练。

### 训练集规则

每题原始 3 个测试中：

- 前 2 个作为公共测试，用于强化学习训练奖励；
- 至少保留 1 个原始测试作为内部隐藏测试；
- challenge tests（若存在）全部进入隐藏测试。

监督微调只使用 train split 的参考实现。

### 验证/测试规则

validation / test 只暴露 1 个公共测试，其余测试保留为隐藏测试。

注意：CodeRL-Lab 的生成提示默认**不直接包含测试内容**。公共/隐藏的区别主要用于奖励可见性和评测隔离。

## 第二层：MBPP+ —— 强化隐藏测试评测

MBPP 原始测试数量太少，容易把脆弱代码误判为正确。

EvalPlus 的 MBPP+ 为原 MBPP 增加了大约 35 倍测试，因此用于：

- 检查模型是否只过了少量原始测试；
- 比较 SFT / GRPO 后的鲁棒正确率；
- 检查公共测试奖励是否造成测试过拟合。

MBPP+ **不用于训练奖励**。

## 第三层：LiveCodeBench —— 外部分布评测

LiveCodeBench 按发布时间持续收集 LeetCode、AtCoder、Codeforces 等竞赛问题，并专门提供按时间窗口评测能力。

它用于后续：

- 外部分布泛化；
- 基础模型与后训练模型的相对比较；
- 污染敏感分析；
- 能力边界分析。

当前不把 LiveCodeBench 用作第一轮训练数据，避免把执行框架复杂度和强化学习问题混在一起。

## 污染问题

MBPP / HumanEval 都是长期公开基准，Qwen3 等现代基础模型可能在预训练阶段接触过其内容。

因此 CodeRL-Lab 明确禁止以下结论：

> “模型在 MBPP 上提升，所以强化学习创造了全新的代码能力。”

MBPP 阶段只能回答：

- 相同基础模型下 SFT 与 RL 的相对变化；
- Pass@k 分布如何变化；
- 公共/隐藏测试差距如何变化；
- 动态采样是否提高训练效率。

真正的能力边界结论必须结合更独立、更新或时间可控的外部评测。

## 数据可复现

原始数据不直接复制进 Git 仓库。

GitHub 保存：

- 构建代码；
- 数据源 ID；
- 数据源解析后的 commit SHA（若可获取）；
- Hugging Face split fingerprint；
- 输出文件 SHA-256；
- split 计数；
- 转换配置。

Tang 下载和转换后，只把 `manifest.json` 的精炼信息写回 GitHub。

## 生成文件

运行：

```bash
python -m coderl_lab.datasets.mbpp \
  --output-dir data/generated/mbpp-v1
```

生成：

```text
train_tasks.jsonl
train_sft.jsonl
validation_tasks.jsonl
test_tasks.jsonl
manifest.json
```

其中生成的数据文件不提交 Git；manifest 会进入实验记录。
