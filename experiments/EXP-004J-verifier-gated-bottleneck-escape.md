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


# 正式 90-task 结果

正式 runner 完成：

- SFT reference reproduction：90/90；
- eligible tasks：57/90；
- public-fail + low-margin gate triggered：36/90；
- high-margin control available：57/90；
- GPU / Docker 结束后无残留任务。

## 全 90 题 hidden correctness

| 条件 | correct | accuracy | 相对 baseline |
|---|---:|---:|---:|
| SFT baseline | 36/90 | 40.00% | — |
| gated low-destabilize | **39/90** | **43.33%** | +3.33 pp |
| gated high-destabilize | 36/90 | 40.00% | 0 |
| always low-destabilize | 38/90 | 42.22% | +2.22 pp |

### gated-low - baseline

- observed delta：+3.33 pp；
- 95% task bootstrap CI：**[0,+7.78] pp**；
- P(delta>0)=0.954；
- wrong→correct：3；
- correct→wrong：0。

因此点估计为正，且没有观察到 correctness harm，但**没有严格通过预注册的“95% CI 下界 >0”强成功标准**，因为下界等于0。

### gated-high - baseline

- delta：0；
- 95% CI：[0,0]；
- wrong→correct=0；
- correct→wrong=0。

相同 gate、相同 -0.25 bias，只把干预位置改到 high-margin token 后效果完全消失，支持位置特异性。

### always-low - baseline

- +2.22 pp；
- 95% CI：[-3.33,+7.78] pp；
- wrong→correct：4；
- correct→wrong：2。

无 gate 的 low-destabilize 虽然也能解锁部分正确轨迹，但会伤害原本正确的任务。

## Gate-triggered 36 题

- baseline：2/36 = 5.56%；
- gated-low：5/36 = 13.89%；
- gated-high：2/36 = 5.56%；
- gated-low delta：+8.33 pp；
- 95% CI：[0,+19.44] pp；
- wrong→correct：3；
- correct→wrong：0。

同样是正向、无观察到伤害，但当前样本量下 CI 下界为0。

## Public-pass 保护

baseline public tests 全通过的任务：37。

- gated-low changed：0/37；
- gated-high changed：0/37；
- always-low changed：21/37。

因此 public gate 达到了预期的安全作用：

> 对 verifier 已认可的 SFT output 不进行 escape，避免无必要扰动。

## EXP-004J 结论

当前证据支持以下**有限**结论：

1. SFT-only verifier-gated low-margin escape 可以产生正向 correctness 点估计；
2. 观察到 3 个 wrong→correct、0 个 correct→wrong；
3. high-margin control 完全无效，说明不是任意二次解码或任意 -0.25 bias 都有效；
4. public gate 能避免对 public-pass 任务的无谓改写；
5. 但是 primary 95% CI 下界为0，因此不能在当前90题上宣称可靠总体提升。

这90题已经被 EXP-006A/B、004F-I 多次用于机制开发，继续在同一 validation 上调整阈值/bias 会增加研究者自由度和过拟合风险。

## 下一步：冻结规则，做 held-out replication

不再调整：

- low threshold = 0.05；
- high threshold = 0.20；
- bias = -0.25；
- primary window = 128；
- public fail gate；
- deterministic greedy。

直接在此前未参与方法设计的 MBPP test split 500 tasks 上复现。

只有 held-out test 仍显示 gated-low > baseline，才能把 EXP-004J 从机制 proof-of-concept 推进为更可信的 decoding intervention 结果。
