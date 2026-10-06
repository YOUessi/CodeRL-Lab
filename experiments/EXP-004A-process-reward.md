# EXP-004A：执行阶段可验证奖励 / 代码完整性信用分配

## 状态

- 奖励设计：已实现第一版
- 静态依赖分析：已修复嵌套作用域误报
- 固定输出离线诊断：已完成
- 1.7B SFT 全训练集 4-sample 奖励方差审计：运行中
- 2-step GPU 冒烟：待离线审计通过后运行
- 187-step 正式训练：未开始

## 研究动机

EXP-006B 的大 k 分析表明，大多数所谓能力丢失可以被更大采样预算恢复。
唯一长期 0/64 的 mbpp_validation_0534 进一步发现：

- SFT 60/64 候选会使用 re.*；
- 但 60/60 缺少 import re；
- 只补 import re 后 17/64 直接通过隐藏测试。

因此当前具体问题不是模型完全不会算法，而是模型知道核心操作但输出代码不完整，导致运行阶段失败。

## 对照原则

基线：EXP-006A 的 1.7B 纯 GRPO。

唯一主变量：

旧奖励：语法 + 公共测试 + 全通过奖励。
执行阶段奖励：语法 + 依赖完整性 + 运行时干净 + 公共测试 + 全通过奖励。

其它保持不变：

- Qwen3-1.7B-Base 固定 revision；
- 同一个 EXP-006A SFT adapter；
- 同 train split；
- num_generations=4；
- 187 optimizer steps；
- beta=0；
- loss_type=grpo；
- 相同学习率、batch、seed；
- hidden tests 永不进入训练 reward。

## 第一版执行阶段奖励

权重：syntax 0.05，dependency_complete 0.10，runtime_clean 0.10，public_test 0.65，all_public_pass 0.10。

依赖完整性使用 Python symbol table 做作用域分析，避免把 lambda 参数、comprehension 变量、nested helper 参数和递归 helper 误判为缺依赖，同时仍能检测 re、math、Counter 等真实缺失依赖。

运行时干净信号将 AssertionError 视为正常逻辑判定，将 NameError、TypeError、IndexError、timeout、crash 视为运行阶段失败。

## 固定输出离线诊断

在 EXP-006B n=16 的 4320 个 validation completions 上，修复作用域误报后：

Base：hidden-correct dependency complete 100%，hidden-incorrect 56.13%；hidden-correct runtime clean 100%，hidden-incorrect 45.98%。
SFT：hidden-correct dependency complete 100%，hidden-incorrect 74.64%；hidden-correct runtime clean 99.81%，hidden-incorrect 60.24%。
GRPO：hidden-correct dependency complete 100%，hidden-incorrect 77.07%；hidden-correct runtime clean 99.82%，hidden-incorrect 63.11%。

高频 unresolved names 包括 re、math、gcd、heapq、cmath、groupby、Counter。
高频 runtime failures 包括 NameError、TypeError、IndexError、SyntaxError。

因此依赖完整性和运行时干净都与 hidden correctness 存在明显区分力。

## 训练前强制门槛：奖励方差审计

固定 1.7B SFT policy，对全部 374 个 train task 各采样 4 次，共 1496 completions。
对同一批 completions 同时计算旧 outcome reward 和新 execution-stage reward。

重点比较：

- outcome flat-group fraction；
- process flat-group fraction；
- 多少 outcome-flat prompt 被过程奖励变成 mixed；
- 是否有 outcome-mixed 被错误压成 flat；
- rescued flat 主要来自 dependency 还是 runtime 信号。

只有过程奖励能明显减少 zero-std prompt，才进入正式 GRPO。

## 正式训练门槛

离线审计通过后：16 task / 2 step smoke → reward component 日志检查 → hidden-test 泄漏检查 → OOM / Docker 泄漏检查 → 187-step 正式训练。

## 正式评测

统一使用 EXP-006B 口径：MBPP validation 90 题、n=16、Pass@1/4/8/16、hidden mean、solved@16、syntax/runtime/dependency diagnostics、zero-grad / frac_reward_zero_std、配对 bootstrap。

最终回答：更细的执行阶段可验证奖励，能否提高真正隐藏测试正确性与代码完整性，而不仅仅让训练 reward 更密？

## 离线审计预注册通过门槛

为避免看到结果后再调整判断标准，在全训练集 1496 个固定 rollout 完成前预先规定：

- process flat-group fraction 相比 outcome flat-group fraction 至少下降 10 个百分点；
- 或者 process reward 至少救活 20% 的 outcome-flat prompt；
- mixed→flat 的组不能超过全部 prompt 的 5%；
- 审计只使用 train split 公共测试和静态/运行时训练可见信号，不使用 hidden test。

满足前两项任一项、且不违反 mixed-collapse 限制，才进入 2-step GPU smoke 和 187-step 正式训练。
