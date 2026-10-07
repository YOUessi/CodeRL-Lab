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

## verified import-repair 数据审计

来源：1.7B SFT policy 的 1496 个固定 train rollout。

严格规则：只补标准库 import，除 import 前缀外不改算法；补完后必须重新执行 public tests 并 100% 通过；hidden tests 不参与构建。

结果：

- verified pairs：97；
- distinct tasks：56；
- 每任务1对后的正式 v1 数据：56 pairs；
- pair / source prediction：6.48%；
- hidden_tests_used：false。

主要修复类型：

- import re：51；
- import math：16；
- from collections import Counter：7；
- import heapq：6；
- OrderedDict：5；
- defaultdict：3；
- statistics / groupby / deque 等少量。

被过滤：

- 无 unresolved names：1216；
- 补 import 后仍不能全过 public tests：138；
- unresolved name 非标准库白名单：22；
- duplicate pair：23。

这说明大多数 runtime failure 不是“补 import 就能修”，v1 只保留最干净的 97 个近邻样本。

正式训练数据进一步按 task 去重：每个任务只保留 original public pass rate 最高的 pair，tie 时取更小 sample_id，得到 56 对。

## 2-step DPO GPU smoke

设置：16 pair、max_steps=2、beta=0.1、sigmoid DPO、SFT adapter 为 policy 初始化，TRL 自动复制冻结 ref adapter。

结果：

- train runtime：4.01 s；
- grad norm：5.94 / 6.20，非零；
- peak allocated：4.36 GB；
- peak reserved：4.69 GB；
- adapter 保存成功；
- ref adapter 单独保存于 ref/；
- Docker 残留：0。

2-step 中 DPO reward margin 仍为 0：第1步 LR=0，第2步日志在有效更新前记录，因此 smoke 只用于工程/显存验收，不作为学习效果证据。

结论：可以进入 56 pair × 3 epoch 正式 DPO。

## 正式 DPO 训练结果

正式数据：每个任务只保留 1 个最接近正确的 verified pair，共 56 pair / 56 tasks。

训练：

- 初始化：EXP-006A 1.7B SFT adapter；
- beta=0.1；
- sigmoid DPO；
- learning rate=5e-7；
- micro batch=1；
- gradient accumulation=8；
- 3 epoch；
- 21 optimizer steps。

结果：

- train runtime：35.71 s；
- train loss：0.6734；
- peak allocated：4.57 GB；
- peak reserved：5.64 GB；
- output adapter SHA-256：860328992b8190b51ec595c08a2d36ebfecd4801db01fd8159d1bf5a0d96d18f。

### DPO 学习信号

前25% → 后25%：

- reward margin：0.0076 → 0.0579；
- preference accuracy：37.5% → 90.0%；
- DPO loss：0.6895 → 0.6649。

最后 5 个 step 的 preference accuracy 分别为 1.0、0.625、0.875、1.0、1.0。

因此 DPO 已经明确提高 chosen（最小 import 修复）相对 rejected（原 runtime-failure completion）的偏好，不是仅仅完成了训练流程。

当前 90题 × 16 validation 正在运行。最终只有 dependency/runtime 错误下降且 Pass@k 不被明显破坏，才能认为 EXP-004B v1 有效。
