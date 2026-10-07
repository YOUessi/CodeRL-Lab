# EXP-004J：Verifier-Gated Bottleneck Escape

## 状态

- 分支：exp/verifier-gated-bottleneck-escape
- 设计：已预注册
- SFT-only runner：待实现
- 4-task GPU smoke：待运行
- 90-task 正式实验：待运行

## 研究动机

EXP-004I 已证明：

- low-margin bottleneck 控制 trajectory identity；
- 将 candidate stabilize 回 SFT reference 不提高 hidden correctness；
- 当 SFT reference 错误时，low-destabilize +9.21 pp，95% CI [+1.97,+18.42]；
- public-test failure 对 hidden-wrong 的 precision=94.44%、recall=89.47%；
- offline public-gated policy +5.26 pp，95% CI [+0.44,+11.40]。

但 EXP-004I 仍使用了 perturbation candidate adapter。

EXP-004J 移除这个中间变量，直接测试：

> 在 SFT policy 自己上，public-test failure 是否能触发一次 targeted low-margin escape，并提高最终 hidden correctness。

## 固定模型

- Qwen3-1.7B-Base
- revision: ea980cb0a6c2ae4b936e82123acc929f1cec04c1
- EXP-006A SFT adapter
- deterministic greedy
- max_new_tokens=512

不训练任何新参数。

## 第一阶段：SFT baseline

对 MBPP validation 90 tasks：

1. 生成 SFT greedy completion；
2. 计算前 min(128, completion length) token 的 probability top1-top2 margin；
3. 选择第一个 margin<=0.05 的 low bottleneck；
4. 选择离 low bottleneck 最近的 margin>=0.20 high-margin control；
5. 运行 public tests。

hidden tests 在此阶段不参与任何决策。

## Gate

仅当：

- public tests 未全部通过；
- 且存在 low-margin bottleneck；

才触发第二次解码。

否则直接保留原 SFT greedy output。

## 四个条件

### A. baseline

原始 SFT greedy。

### B. gated_low_destabilize

仅 gate 触发时：

- 在 first low-margin bottleneck；
- 对原 SFT top1/reference token 加 -0.25；
- 继续 deterministic greedy。

否则输出 baseline。

### C. gated_high_destabilize control

相同 gate、相同 -0.25 bias。

唯一改变：

- 干预位置为最近 margin>=0.20 的 high-margin token。

用于检验效果是否具有 bottleneck position specificity。

### D. always_low_destabilize

所有存在 low-margin bottleneck 的 task 都执行 -0.25，
不看 public tests。

用于检验 public gate 是否通过避免正确任务上的不必要 escape 来提高净收益。

## Primary outcome

全 90 task hidden-test exact correctness。

主要比较：

1. gated_low - baseline；
2. gated_low - gated_high；
3. gated_low - always_low。

统计：

- task paired bootstrap；
- 20,000 iterations；
- seed=42。

## Transition counts

每个条件报告：

- wrong->correct
- correct->wrong
- correct->correct
- wrong->wrong

不能只报告平均准确率。

## 预注册成功标准

强支持 verifier-gated bottleneck escape：

1. gated_low - baseline 的 95% CI >0；
2. gated_low - gated_high 点估计为正，优先要求 CI >0；
3. wrong->correct 明显多于 correct->wrong；
4. public-pass task 因 gate 不触发，不产生额外伤害。

如果 gated_low 不优于 baseline：

说明 EXP-004I 的正向结果依赖 perturbation candidate context，不能直接转成 SFT-only decoding 策略。

如果 gated_low 与 gated_high 相近：

说明效果不是 low-margin position-specific，而可能只是第二次扰动/重生成效应。

## 资源与记录

- 不训练新模型；
- GPU 仅做 deterministic generation；
- Docker public/hidden evaluation 串行与 GPU generation 分离；
- 所有代码与记录先进入 GitHub；
- 当日过程写入 docs/daily/2026-10-08.md；
- 失败和负结果同样保留。


## 4-task GPU smoke

代码与测试：

- 139/139 CPU tests passed；
- SFT reference reproduction：4/4；
- eligible tasks：2/4；
- gate triggered：2/4；
- high-margin control available：2/4。

Smoke tasks 中：

- mbpp_validation_0511：public fail，low bottleneck token101，low-destabilize 正常应用并改变 greedy trajectory；high-margin token100 的 -0.25 被应用但完整输出保持 baseline；
- mbpp_validation_0512：public fail，low bottleneck token17，low-destabilize 正常应用并改变 trajectory；high-margin token16 的 -0.25 不改变完整输出；
- 0513 / 0514：无 low-margin bottleneck，因此 gate 不触发，所有 gated 输出保持 baseline。

4 个 smoke task 的 hidden correctness 都为0，因此 smoke 只验证工程逻辑，不作为有效性结论。

Smoke 结论：

- reference reproduction 正常；
- public gate 正常；
- low/high position 干预逻辑正常；
- public-pass/no-bottleneck 情况不会被误改写；
- 允许进入正式 90-task。

## 正式 90-task 运行

正式参数保持预注册不变：

- low threshold=0.05；
- high threshold=0.20；
- bias=-0.25；
- deterministic greedy；
- max_new_tokens=512；
- public tests 只用于 gate；
- hidden tests 只做最终 outcome；
- GPU 串行，不并发第二个生成任务。

正式 runner 已在 Tang 启动，完成后自动进入 Docker hidden-test evaluation 与 20,000 次 paired bootstrap。
