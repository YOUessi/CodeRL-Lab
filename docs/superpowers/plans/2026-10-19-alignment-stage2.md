# CodeRL-Lab 对齐与强化学习第二阶段实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 完成已确认的展示、奖励诊断、PPO契约与有限预算7B研究闭环。
**Architecture:** 默认首页只增加入口；PPO与7B科研保持独立分支。分析、训练、生成、评分有独立入口与可核验中间产物，GPU由主协调者顺序调度。
**Tech Stack:** Python、PyTorch/TRL/PEFT、Docker、pytest、GitHub Git Data API、Tang CUDA。
**Spec:** ../specs/2026-10-19-alignment-stage2-design.md

## Global Constraints
- GitHub 是代码、配置、过程和结果的唯一事实源；代码提交后才部署 Tang。
- 三种子51/52/53，每种子最多128步，保存32/64/128，主比较128−32。
- 保留固定SFT、Stage1标签与全部历史 artifacts。
- 新最终评测集合、执行资源和主指标在训练前封存；最终题不用于调参。
- 独立任务允许并行；单GPU只能有一个训练/生成作业。
- 任何新实验的成功与失败都保存，不保证分数上涨。

## Review Focus
- 模型/SFT/数据哈希与运行配置不一致时，必须在训练前失败。
- 已有输出、半完成快照、截断JSONL不能被覆盖或默认为成功。
- EOS、长度截断与padding的边界必须与目标和mask一致。
- 重复任务/采样位置、隐藏测试提前读取、参考解失效必须被检查。
- 相同训练前缀、优化器/RNG/采样状态与恢复语义不能仅靠适配器文件推定。

## Task 1: 默认成果入口
**Files:** main: README.md；docs/experiment-logs/2026-10-19-alignment-delivery.md。
**Interfaces:** 消费冻结报告链接；输出三个可发现入口，保持旧正文历史性。
- [ ] 获取main最新head并确认固定报告目标。
- [ ] 在顶部添加7B研究、方法索引、复算入口及明确结果边界。
- [ ] 校验链接和diff，按expected_sha提交；不合并其他代码分支。

## Task 2: 历史奖励诊断
**Files:** 新增 src/coderl_lab/analysis/reward_signal_audit.py、对应tests、按日期报告与results。
**Interfaces:** 消费显式artifact根下EXP028 v3奖励/任务日志；输出无程序正文、无测试正文的逐组分类及按族汇总。
- [ ] 用合成日志验证全对/全错/相同中间分/非零差异，缺项、非法数值、重复组应明确失败。
- [ ] 实现只读CLI；报告分组规则和原文件哈希。
- [ ] 提交GitHub后在Tang原产物上运行，提交实测结果和解释。
- [ ] 独立复核统计与既有28/32、30/32、19/32、19/32证据一致。

## Task 3: PPO实现契约
**Files:** PPO独立分支 src/coderl_lab/train/ppo_online.py、ppo_math.py、必要独立contract模块、对应tests、中文报告。
**Interfaces:** 保持可运行PPO入口；显式记录终止类型、有效token、行为概率定义和GAE末端语义。
- [ ] 复读实际实现和smoke，选择并记录有限时域目标与采样策略契约。
- [ ] 以确定性算例检查EOS、截断、mask、GAE和概率比；先确认缺陷再最小修复。
- [ ] 针对变更验证，提交固定来源和测试证据；不将CPU算例称新增GPU效果。
- [ ] 独立审查修复与历史记录的边界。

## Task 4: EXP030可配置训练入口
**Files:** 新增 src/coderl_lab/train/exp030_budget_grpo.py、configs/exp030_budget.yaml、对应tests。
**Interfaces:** CLI接受config、seed、output-dir和显式资产路径。每个运行输出identity.json、summary.json或failure.json、reward_attempts.jsonl，以及checkpoint-32/64/128。
- [ ] 核对现有训练回调和执行层，优先复用稳定奖励实现。
- [ ] 实现参数/资产预检、不可覆盖目录、128步硬上限、共同轨迹检查点与完整训练状态保存。
- [ ] 验证错误路径、检查点边界、日志计数与配置；明确恢复支持范围。
- [ ] 独立审查并提交；由主协调者部署Tang运行。

## Task 5: 预注册、独立评测与运行调度
**Files:** experiments/EXP-030-budget-preregistered.md；新评测manifest、生成/评分入口和测试；日期运行日志。
**Interfaces:** 公开prompt manifest与隐藏测试句柄分离；生成冻结清单绑定模型和输出哈希，评分仅消费已冻结答案。
- [ ] 清点本地来源与既往评测任务身份；固定新的开发/最终题目集合和来源revision。
- [ ] 冻结预注册：主指标候选正确率、次指标固定首样本/四次覆盖/错误/成本，按题配对统计。
- [ ] 预检Tang显存、磁盘、模型、SFT、镜像和参考解协议。
- [ ] 串行运行三个种子128步，保留32/64/128及完整训练日志。
- [ ] 冻结SFT与六个32/128模型的最终答案，之后统一判题；报告所有种子。
- [ ] 达预算即停止，不根据最终分数加跑；故障按同协议恢复或如实报告停止。

## Task 6: 验收与展示
**Files:** EXP030最终报告、公开逐题数据、运行记录、能力索引与README导航。
- [ ] 复核模型/产物身份、参考解、任务去重、统计口径及训练成本。
- [ ] 验证新增测试与受影响原有测试；检查GitHub CI。
- [ ] 独立审查全部结果和结论，修复实质问题。
- [ ] 提交最终报告，给出全部分支/提交/结果链接与实测边界。
