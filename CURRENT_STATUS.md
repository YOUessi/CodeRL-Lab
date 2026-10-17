# CodeRL-Lab 当前阶段｜2026-10-18

## 已收尾的支线
异常分类修复支线至EXP-027收尾。修复作为工程正确性改进保留，不再重复相同配置追求正结果。统一执行观测后，两个seed中旧严格/修复版的奖励、最终模型和同题评分均相同；不能把此前有超时混杂的正向变化归因于奖励改进。

历史完整记录：`docs/experiment-logs/2026-10-18-exp027-matched-execution-final.md`。

## 当前主线：EXP-028 训练数据覆盖
主模型固定 **Qwen2.5-7B-Instruct + 同一7B SFT adapter**，不退回0.6B/1.5B。

- narrow：原6个算法族，各8条，48条。
- broad：12个算法族，各4条，48条；其中24条与narrow完全共享，24条来自新增6族。
- 同一corrected奖励、同一受保护执行协议、相同32步/4候选/256token预算、seed43/44。
- 首轮只改变算法族分布，仍保持solve(x)返回整数。多函数接口和任意返回类型扩展不是本次已实现内容。
- 同样任务数与token上限不代表真实耗时相同，实际预算使用单独计数。

协议及停止条件：`experiments/EXP-028-training-coverage.md`。
实施计划：`docs/superpowers/plans/2026-10-18-exp028-coverage.md`。

## 实际执行入口
Tang执行目录：`/home/you/projects/CodeRL-Lab-exp028-pinned`。
产物目录：`/home/you/projects/CodeRL-Lab-track-train017/artifacts/exp028`。
代码从GitHub固定提交获取。普通git fetch超时后改用GitHub已认证归档/Contents API；已逐文件核对411个跟踪文件的Git blob SHA，与提交`0cfeb84caa62a4491f450a851234db8fab08c53c`全部一致，不依赖旧工作树的未提交文件。

连续流程：参考解验证 → SFT训练前检查 → 固定门槛 → 四次训练 → 全部评测答案冻结 → 同一协议评分 → 配对统计 → GitHub安全汇总。

入口：`scripts/run_exp028_chain.py`。门槛失败会停止并写原因，不自动换题或加训练；任何无法可靠判分异常不记零，停止后续。已有进程运行时不能重复启动。

截至本文件创建，数据已冻结、72个参考解通过真实Docker、本地完整238项测试通过，SFT检查正在运行；新的训练和评测结果尚未产生。后续以Tang的`state.json`、`gate_report.json`、`final.json`及本分支`docs/experiment-logs/2026-10-18-exp028-coverage-result.md`为准。不存在最终文件就不能称该阶段完成。

独立评测冻结的是排除历史216题后的新96道MBPP+，不是跨数据集验证，也不能证明无预训练污染。旧96题不再用于选择本轮方案。
