# 2026-10-18｜EXP-028 v2 停止状态核验

## 当前结论
本次于 Tang 2026-10-18 16:29（北京时间）核查真实产物与进程。EXP-028 v2 已停止，不是仍在排队或训练；没有外部评测结果，不能判断宽覆盖是否改善编程能力。

停止时间来自 state.json：2026-10-17T21:43:07.213511+00:00，即北京时间 2026-10-18 05:43:07。状态 stage=failed / status=failed。这里只补记核查，不修改或覆盖历史运行。

## 分阶段结果
| 阶段 | 状态 |
|---|---|
| 训练前门槛 | 已通过；窄组38/48个差异奖励组，宽组26/48，新题8/24个差异奖励组 |
| narrow seed43 | 32/32步完成；summary与adapter SHA一致 |
| broad seed43 | 第一组判题发生无法可靠判分的墙钟超时；0个已完成优化步 |
| narrow seed44 | 未启动，没有运行目录 |
| broad seed44 | 未启动，没有运行目录 |
| 新冻结96题完整评测 | 未执行；final.json和github_delivery.json不存在 |

narrow seed43 相对SFT变化参数元素1,852,469；adapter SHA256 d17670683ae80cb14a8a9f60fd1cc81fc71c768fa2592a878e3d4ea81079b1b3。重新计算文件SHA与summary保存值一致。

## 直接失败证据与边界
`broad_seed43/unscorable_abort.json`：
- error_type=SandboxIndeterminate
- reason=wall_guard_expired_no_reward
- group_id=0
- reward_group_committed=false
- 四个task_id均为 train/gcd_target/1

`train_broad_43.log` 显示 exp028_resource_guard 的 Docker调用触发宿主60秒墙钟保护，错误向上抛出，GRPO在0/32进度退出；`chain_failure.json`记录 no_automatic_retry=true。本次不是前一轮已经处理的已确认OOM分类事件，也不是GitHub CPU单元测试失败。

第一组是两臂共用的旧题型，不能据此认定新增题型错误或宽覆盖无效。该运行目录仅保留 identity.json 与 unscorable_abort.json，没有 reward_attempts.jsonl 或 local_candidates.jsonl；失败组未提交。现有持久化记录能证明超时发生，但不能唯一定位至某份生成程序，也不足以区分候选死循环、算法过慢或运行时调度开销。

## 当前运行与CI
核查 /proc 中实际Python/脚本进程，未发现活动EXP-028任务。Tang RTX4090 Laptop显存324MiB、利用率1%，处于无本项目计算任务状态。按coderl命名前缀查询未发现残留容器；此查询不是对所有Docker服务的全面审计。

本次状态文档提交之前的最新GitHub CI：39071920301，提交4c1c61290e17348865b159e801890548049ffe71，completed/success。CI通过不代表训练已完成。本次不改训练代码、数据、种子、奖励或预算，不自动重跑失败组。

## 后续处理约束
先补全失败候选的持久化诊断与执行预算分类，再决定受控恢复方式；不能仅提高时间限制、删除失败标记、跳过题目或将宿主超时记作零奖励来让链路继续。若需要改变执行协议，必须单独记录修订，并对两臂使用相同条件。尚无证据支持宽覆盖正收益或负收益。

原始产物：/home/you/projects/CodeRL-Lab-track-train017/artifacts/exp028_v2/。训练测试与候选正文未上传。