# 2026-10-19｜大模型对齐与强化学习现状核查

## 任务范围和既有决定

用户要求优先完成大模型对齐与强化学习方向，并检查CodeRL-Lab现状。衔接2026-10-18已接受的阶段交付范围：总报告、README、复现入口、稳定研究快照；暂停新增训练。

本次核查以GitHub为来源，按固定提交读取；并行核查7B研究证据及其他对齐方法分支，随后在Tang实际只读验证现有发布目录和原始产物。没有把早期main首页当作当前7B进展，也没有重复启动已完成的实验。

## 核查基准

- 研究标签：`stage1-7b-v1.0.0`。
- 标签与核查时发布分支提交：`dac5bcf04d5d759b5baa616ec89f68e12316440d`。
- 实际Git tree：`b3a48231058dd853fab314c69394872adce19b8d`。
- Tang发布目录：`/home/you/projects/CodeRL-Lab-stage1-release`。
- 原始产物根：`/home/you/projects/CodeRL-Lab-track-train017/artifacts`。
- 设备：RTX4090 Laptop GPU，16376MiB。最初检查时约324MiB占用、1%利用率；未发现本项目训练进程。这是该时点观测，不代表持续监控。

## 实际执行及结果

| 执行 | 结果 |
|---|---|
| 对发布目录448个Git跟踪文件逐个计算blob标识，与固定GitHub树核对 | 448/448一致，无缺失或内容差异 |
| `python3 -S scripts/stage1_release.py demo` | 退出0；五模型correct为202/202/201/203/207，分母均384 |
| `python3 scripts/stage1_release.py verify --bootstrap` | 退出0；公开计数与10份文档目标检查通过，NumPy2.2.6下20,000次按题配对复算区间一致 |
| `python3 -S scripts/stage1_release.py audit --artifact-root /home/you/projects/CodeRL-Lab-track-train017/artifacts` | 退出0，`OWNER_ARTIFACTS_VERIFIED_READ_ONLY` |
| `python3 -m pytest -q -p no:cacheprovider`，显式`PYTHONPATH=src`并禁写bytecode | 退出0，291 passed in 3.56s |
| 标签CI 39137698707 / 原发布提交CI 39137446692 | 两者均completed/success |

原始审计核查1920个采样位置、96个参考解记录、4个训练模型文件；源final.json SHA256为`cffb5075c0b07998762a32e1b83906b2b1a07c704b619e97cb4f9355c8e9ccd4`。公开CSV SHA256为`b19ef3940366f7fb838ad4d3965f17329e50669d5461190f70ad55e3b1fbe9b9`。

这些是文件、记录、统计与来源一致性检查。本次没有执行新的GPU训练、模型生成或候选容器判题；未打印生成正文或读取隐藏测试正文。其他对齐分支的历史训练结果依据其固定代码和结果汇总，本次没有逐一重读其模型权重。

## 核查发现与收尾修改

1. 7B主线已完成至EXP-029，主要数值和统计边界未发现实质错误；没有确认稳定外部编程收益。不能沿用早期讨论中的“seed44尚未完成”作为当前状态。
2. 真实7B SFT/DPO、1.7B奖励模型/PPO工程试跑存在于其他分支，稳定版没有全部收录。增加[对齐与强化学习成果总览](../reports/ALIGNMENT_RL_STATUS_ZH.md)，给固定提交来源、实际完成度与限制，并从发布分支README链接。
3. EXP-015已将DPO旧评分122/245更正为与训练器一致口径的121/245，三模型相同。本次入口使用更正后的回答总对数似然排序指标，不误称隐式奖励或生成质量增益。
4. EXP-026历史诊断报告§5的“只改依赖分类”容易单独读成有效因果比较。仅新增日期明确的阅读提示，指向后续超时混杂更正和EXP-027，保留全部历史数据。
5. 其他对齐方法仍有研究缺口。PPO只有1.7B工程试跑，单AI盲审的真人审阅者为0；不把这些写成7B正式PPO或真实双人盲审。

本次修改仅涉及成果入口和说明。稳定标签不移动；研究代码、模型、奖励定义、测试集合、公开计数和原始产物不因本次核查改变。main及其他实验/界面分支不自动合并。

## 遇到的问题及处理

当前审计工作区通过HTTPS克隆时，网络代理端口不可达。没有修改代理、凭据或安全设置；改用GitHub连接器按固定提交读取，利用Tang已有独立发布目录进行源文件一致性核验与CPU/只读验证。

Tang历史实验worktree显示detached HEAD及未跟踪研究脚本，不能据该工作树HEAD推断实际执行源码。本次改为对独立发布目录逐文件校验448个Git blob；未清理、覆盖或切换历史工作树。

## 证据入口

- [当前能力与固定分支来源](../reports/ALIGNMENT_RL_STATUS_ZH.md)。
- [第一阶段总报告](../reports/STAGE1_REPORT_ZH.md)。
- [复现分层与命令](../reproduction/STAGE1.md)。
- [标签CI](https://github.com/YOUessi/CodeRL-Lab/actions/runs/39137698707)。

本文件记录的CPU/原始产物验证基于固定标签提交；本次文档提交的最终CI以对应GitHub Actions运行记录为准，不预填未发生的验证结果。
