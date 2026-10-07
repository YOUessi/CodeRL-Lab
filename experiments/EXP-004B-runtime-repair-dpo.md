# EXP-004B：局部运行时修复偏好学习

## 状态

- 偏好对构建器：已实现第一版
- 最小 import 修复数据审计：待运行
- DPO：待数据审计通过后实现
- 正式训练：未开始

## 动机

EXP-004A 表明：

- 更细的执行阶段 reward 能显著减少 zero-grad / zero-std；
- dependency incomplete 与 runtime failure 也略有下降；
- 但 MBPP n=16 与 MBPP+ 均没有能力增益证据。

因此不再继续调 reward shaping。

EXP-006B / EXP-004A 已定位到具体 failure mode：

- 候选知道应使用 re / math / Counter / heapq 等；
- 但缺失 import，导致 NameError；
- 这类错误与算法主体无关，适合做局部偏好学习。

## v1：只做“标准库 import 最小修复”

数据来源：

- 1.7B SFT policy 的 374 train tasks × 4 rollout；
- 1496 个固定 candidates；
- EXP-004A 已生成的 dependency/runtime diagnostics。

只接受满足全部条件的 pair：

1. rejected 原候选公共测试未全通过；
2. unresolved names 全部属于标准库白名单；
3. chosen 只是在 rejected 顶部补必要 import；
4. 除 import 外不改算法主体；
5. chosen 在 train public tests 上 100% 通过；
6. hidden tests 完全不参与 pair 构建。

格式：

prompt + chosen + rejected。

## 为什么先做数据审计

很多 NameError 并不是 import 问题：

- 未定义项目 helper；
- 算法主体错误；
- re 虽缺 import，但使用方式本身也错。

这些样本不能强行变 preference pair。

因此只有“最小 import 后公共测试直接全通过”的样本才进入 v1。

## 数据门槛

进入 DPO 前至少要求：

- verified pair >= 20；
- distinct task >= 10；
- chosen/rejected 只存在 import 前缀差异；
- hidden_tests_used = false；
- 每个 chosen 都重新执行 public tests 并全通过。

若数量不足，不扩大到模糊自动修复，而先重新评估实验方向。
