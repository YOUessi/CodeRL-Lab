# MBPP-v1 数据验证记录

## 状态

**构建器：通过。**  
**完整数据转换：通过。**  
**固定 revision 复现：通过。**  
**Docker 断言执行：通过。**

## GitHub 分支

`feat/formal-dataset-v1`

首次完整构建对应代码提交：

`43a4d5b70d4930ebd563230a9f64900d3bf42498`

## 数据源

- 数据集：`google-research-datasets/mbpp`
- 配置：`full`
- 固定 revision：`4bb6404fdc6cacfda99d4ac4205087b89d32030c`

分割 fingerprint：

- train：`5cdc16311fd8e220`
- validation：`8f9deacfc72ae133`
- test：`44206dd27b0bf01d`

## 分割

| 分割 | 任务数 | 用途 |
|---|---:|---|
| train | 374 | SFT + RL |
| validation | 90 | 调参 / 奖励调试 |
| test | 500 | 同分布最终评测 |
| prompt | 10 | 不进入训练 |

## 测试可见性

训练集每题：

- 2 个原始测试 → 公共奖励测试；
- 至少 1 个原始测试 → 内部隐藏测试；
- challenge tests → 全部隐藏。

验证 / 测试集每题：

- 1 个原始测试 → 公共测试；
- 其余测试 → 隐藏测试。

生成提示中默认不直接提供测试代码。

## 代码验证

### 单元测试

新增断言式 benchmark 执行、setup code、MBPP 转换和泄漏检查后：

```text
24 passed
```

### 5 题真实 MBPP 冒烟

从真实 train split 取前 5 题：

- 转换成功：5 / 5；
- 使用参考实现作为候选；
- 通过 Docker 同时执行公共和隐藏断言；
- Pass@1 = 1.0；
- BAD_TASKS = 0。

这验证了：

```text
MBPP原始记录
→ 入口函数识别
→ starter signature
→ assertion test
→ setup code
→ CodeRL-Lab task
→ Docker executor
```

整条链路可执行。

## 完整转换

生成：

| 产物 | 行数 | SHA-256 |
|---|---:|---|
| train_tasks.jsonl | 374 | `52f92e7d9e38c91fa25c283dcea53a507f3ea0983d6d82b9c3f91fa2a72ce689` |
| train_sft.jsonl | 374 | `0b1fc1a0a09b32e49b38c87caf29b223a761d8b184ebc62ebfb5282b35962d69` |
| validation_tasks.jsonl | 90 | `5317393d970c5d009d726b184b6ac737715d692fadb9f0fb9571972e4307b751` |
| test_tasks.jsonl | 500 | `04d992f8b1519e8457ef985e605ab832078283f79dd181fd0a3e2d219c4a148c` |

## 固定 revision 复现检查

第一次使用远端 `main` 解析出的 commit 为：

`4bb6404fdc6cacfda99d4ac4205087b89d32030c`

随后将该 commit 固定进代码和配置，并重新完整构建。

四个产物：

```text
train_tasks.jsonl      IDENTICAL=1
train_sft.jsonl        IDENTICAL=1
validation_tasks.jsonl IDENTICAL=1
test_tasks.jsonl       IDENTICAL=1
```

即固定 revision 后的结果与首次构建逐字节一致。

## 当前结论

MBPP-v1 已满足进入 EXP-002 监督微调的工程条件：

- 数据源固定；
- 训练 / 验证 / 测试分开；
- SFT 答案只输出 train；
- 公共 / 隐藏奖励测试分开；
- 数据产物可由脚本重建；
- 数据产物哈希可核验；
- Docker 执行兼容 MBPP assert 测试。

## 仍未完成

MBPP 是长期公开数据，不能单独承担“无污染能力边界”结论。

后续必须继续接入：

1. EvalPlus MBPP+：更强隐藏测试；
2. LiveCodeBench：外部分布 / 时间敏感评测。

这两项不会阻塞第一轮 SFT 学习实验，但会阻塞最终科研结论。
