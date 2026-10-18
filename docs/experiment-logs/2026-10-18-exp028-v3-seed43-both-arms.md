# 2026-10-18｜EXP-028 v3 两臂seed43训练均完成

Tang 2026-10-18 16:58（北京时间）检查：窄组与宽组seed43都完成32/32步；执行链已自动进入窄组seed44，最近核验10/32步。宽组seed44待同一执行链接续。无chain_failure、gate_failure或unscorable_abort。此文件是阶段性状态，不是最终评测结果。

## 重新验收的两个模型
| 指标 | 窄组43 | 宽组43 |
|---|---:|---:|
| 实际优化步数 |32|32|
| 奖励记录 |128|128|
| 相对SFT变化参数元素 |1852469|1845239|
| 完整持久化候选组 |32|32|
| 奖励就绪组记录 |32|32|
| 确认CPU预算失败 |0|1|

窄组adapter SHA256：d17670683ae80cb14a8a9f60fd1cc81fc71c768fa2592a878e3d4ea81079b1b3。
宽组adapter SHA256：904fae866044fe1bd6810cac441e566860dddbc60209dc9796fccb96c65681dd。

两个adapter文件SHA和reward_attempts文件SHA均与其summary逐项重新校验。均为real_GPU_GRPO_completed；不是将v2模型复制到新目录冒充重训。

宽组首组已跨过旧v2停止的位置：执行前保存四份候选，明确CPU预算失败的候选按v3资源协议记0，其他三份正常完成，整组正常进入优化。新增CPU软15秒/硬16秒是公开协议修订；两臂同样使用256MiB、1CPU、60秒宿主保护。旧v2记录和模型未覆盖。

全部253项本地测试通过；207份活动Python源码与固定GitHub提交d5e9f870333b28bf12c1e5fdd675ff5a0340070a逐字节对应。运行代码CI 39112816742与前一份验收文档CI 39113462511均success；本次文档不修改运行代码。

尚无新冻结96题的模型正确率结果。后续剩余两次训练、生成冻结、统一评分、配对统计和GitHub摘要由已经运行的scripts/run_exp028_resource_v3.py依次执行。另设置每小时核查、最多12次，完成或失败报告后停止核查；不重复启动任务或擅自改变预算。

当前产物：/home/you/projects/CodeRL-Lab-track-train017/artifacts/exp028_v3/。独立的两模型校验摘要：seed43_acceptance.json。
