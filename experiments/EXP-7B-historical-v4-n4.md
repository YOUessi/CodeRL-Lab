# EXP-7B：7B 模型历史 v4 多次采样与可达性审计

**状态：2026-10-13 冻结协议；全量 GPU 及 private score 尚未产生结果。**

- 模型：Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`，NF4 double quant，FP16。
- 数据：固定 LiveCodeBench v4 101 题公开视图 SHA256 `62239669ec50b2ebee8cad1a0561e072e293a76a9a130d0b214bdbcedf980853`。
- 任务构成：101 道、AtCoder 65 / LeetCode 36、Easy 22 / Medium 34 / Hard 45；与已使用 v5/v6 去重。
- 公开提示采用模型原生 Instruct 模板，固定“只输出代码”、greedy smoke 后转为每题 **4次独立随机采样**。
- 采样规则：temperature=0.8、top_p=0.95、max_new_tokens=512；对题目 ID 和 sample index 导出的固定种子；不因公开测试或先前 v4/v5/v6 隐藏成绩改参数。
- 先把 404 条原始代码和 SHA256 存在 Tang 本地，生成期间只能看公开测试。模型输出不做按题人工修复，不把 private tests 放入回传 prompt。
- 生成完整后验证 checkpoint SHA、404/404 样本、官方仓库 SHA `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` 和 v4 源数据 SHA `d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da`，在保存冻结审计后用官方 Docker 评测。
- 主要描述指标：每题4次是否至少有1个通过 **隐藏测试** 的程序（solved@4）、其占101题比例、404样本级正确比例。bootstrap 20,000 次，按题重采样，seed=42。
- 同时报告公开测试 solved@4、candidate resource-limit、timeout、难度/平台分层。对任意错误统一固定 timeout6s/memory1g，不调整测试标准。
- **研究解释限制：**这是一批已经使用过的历史 v4 题，不是最新时间前瞻 holdout；7B Instruct 与1.7B Base使用不同 prompt，不得把两者差异称为参数规模的严格因果效应。
- 本研究首次连续生成 + 官方 private 判分入口：`scripts/run_exp7b_n4.sh`；失败与负结果必须记录并提交。
