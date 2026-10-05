# Results

本目录只保存适合进入 Git 的**精炼实验结果**，例如：

- `summary.json`
- 指标 CSV
- 小型图表
- 失败类型统计
- 实验摘要

以下内容不得提交：

- 模型权重；
- checkpoint；
- Hugging Face 缓存；
- 完整大规模 rollout；
- 原始超大训练日志。

每个结果目录必须能够追溯到 `experiments/EXP-xxx-*.md` 中记录的 Commit SHA 和配置。
