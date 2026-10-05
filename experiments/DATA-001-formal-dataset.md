# DATA-001：正式数据协议与 APPS 接入

## 状态

**GitHub 实现：进行中。**  
**真实 APPS 数据验证：尚未执行。**  
**对应 PR：#3。**

## 目标

在 EXP-002 监督微调之前，把 CodeRL-Lab 从三个内部函数夹具升级到能够承载正式代码训练数据的任务协议。

核心原则：

1. 不为了迁就旧执行器而只保留函数题；
2. 公共测试与隐藏测试严格分离；
3. 原始格式不能无歧义解析时宁可跳过，不静默猜测；
4. 所有过滤、拆分和跳过原因可复现、可统计；
5. 先看真实 APPS 分布，再冻结正式过滤阈值。

## 数据策略

详见：

`docs/dataset-plan.md`

当前定位：

- APPS：训练池 + 内部留出池；
- EvalPlus：外部稳健性 sanity；
- LiveCodeBench：时间标注外部评测；
- 对 Qwen3 无官方精确训练截止日期，因此不宣称 LiveCodeBench v6 对 Qwen3 “绝对无污染”。

## 协议升级

### 模式 1：函数调用

```text
JSON 参数 → 目标函数 → Python 返回值
```

旧 EXP-001 数据如果不包含 `task_mode`，默认继续解析为 `function`。

### 模式 2：标准输入输出

```text
stdin → 完整 Python 程序 → stdout
```

第一版输出规范化：

- CRLF / CR 统一成 LF；
- 每行删除行尾空白；
- 删除整体首尾空行；
- 不改变中间行结构；
- 不做语义模糊匹配。

## APPS v1 适配规则

### 最少测试数

当前暂定：

`min_tests = 4`

少于 4 个有效测试的任务不进入正式 RL 主任务池。

### 公共测试数量

- 4–5 个测试：1 个公共；
- 6–9 个测试：2 个公共；
- 10 个及以上：20% 公共，至少 2 个；
- 其余全部作为隐藏测试。

公共测试索引由：

```text
SHA256(seed + task_id)
```

派生局部随机数，因此与 Python 进程哈希无关，可跨机器复现。

### 明确拒绝的情况

第一版不会猜测：

- input/output 数量不一致；
- 测试数量不足；
- 标准输入不是字符串或单字符串列表；
- 标准输出不是字符串或单字符串列表；
- 没有可用参考 Python 解答；
- input_output / solutions 不是合法 JSON；
- 缺失 problem_id / question。

所有跳过原因必须进入转换报告。

## 当前 GitHub 代码

已实现：

- `CodeTask.task_mode`；
- `FunctionTestCase`；
- `StdIOTestCase`；
- 双模式执行器；
- 双模式提示词和输出归一化；
- `reference_solutions`；
- `coderl_lab.dataset.apps`；
- 双模式可信夹具；
- APPS 假数据单元测试；
- 50 题真实 APPS 转换脚本。

## 真实数据接入日志

### 2026-10-06：Datasets 5.1 不再支持 APPS 旧脚本

第一次在 Tang 执行 APPS 50 题转换时，环境中的 Datasets 5.1.0 报错：

```text
RuntimeError: Dataset scripts are no longer supported, but found apps.py
```

这说明 `codeparrot/apps` 当前仍是旧式数据集脚本仓库，而新版本 Datasets 已不再执行该脚本。

随后通过 Hugging Face Hub 检查仓库真实文件：

```text
revision:
21e74ddf8de1a21436da12e3e653065c5213e9d1

apps.py       4,945 bytes
train.jsonl   107,101,272 bytes
test.jsonl    1,292,436,853 bytes
```

旧 `apps.py` 的作用只是读取 JSONL，并把原始记录中的 `id` 映射成 `problem_id`。

因此决定：

- 不降级整套训练环境；
- 不继续依赖已废弃的数据集脚本执行；
- 固定 APPS revision；
- 直接通过 Hugging Face Hub 下载 `train.jsonl` / `test.jsonl`；
- 自己逐行解析原始 JSON；
- 同时兼容原始 `id` 和旧脚本映射后的 `problem_id`；
- 每条转换任务记录 `source_revision`。

这样数据来源反而更透明、可复现。

## 下一步验证

1. GitHub CI；
2. Tang 拉取 PR #3 分支；
3. 双模式可信夹具分别在 local / Docker 下运行；
4. 从 Hugging Face APPS train 中转换 50 个合格任务；
5. 统计：
   - 扫描多少原始行才得到 50 个任务；
   - function / stdin_stdout 比例；
   - 难度分布；
   - 跳过原因；
   - 原始测试数量分布；
6. 随机抽查转换后的任务；
7. 运行参考解答通过隐藏测试，验证适配语义；
8. 根据真实结果再决定是否调整最少测试数和输出比较规则。

## 合并门槛

- [ ] GitHub CI 全绿；
- [ ] 双模式本地可信执行通过；
- [ ] 双模式 Docker 执行通过；
- [ ] APPS 真实 50 题成功转换；
- [ ] 参考解答执行抽查通过；
- [ ] 现实数据格式和跳过原因写回本文件；
- [ ] v1 数据规则冻结。
