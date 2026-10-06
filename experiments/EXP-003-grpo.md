# EXP-003：可验证奖励 GRPO 基线

## 状态

**环境与接口核验：进行中。**  
**GRPO 代码：待实现。**  
**GPU 冒烟：待执行。**

## 研究目的

在 EXP-002 的监督微调模型基础上加入**可验证奖励强化学习**，建立：

```text
Base
vs
SFT
vs
SFT + GRPO
```

的第一条完整后训练对照。

第一阶段不追求新算法，只要求：

1. 奖励严格只使用 train split 公共测试；
2. 隐藏测试完全不进入训练；
3. GRPO 训练过程可复现、可记录；
4. 与 EXP-002 使用完全相同的 validation Pass@k 流水线；
5. 记录探索、多样性、奖励方差和有效样本比例。

## 初始化策略

从 EXP-002 的 SFT adapter 继续训练，而不是直接从 Base 做 RL。

基础模型：

`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`

SFT adapter：

SHA-256：

`7fcb2b0ca7608be980fb4cbae5158241886ac67f3cc2f5b855cf34531c5982f7`

## 数据

固定 MBPP-v1：

`google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c`

训练时只允许使用：

- train split prompt；
- train split 公共测试。

禁止使用：

- train hidden tests；
- validation hidden tests；
- test hidden tests；
- MBPP+ 隐藏测试。

## 第一版奖励

第一版保持简单，避免奖励函数本身成为主要变量：

```text
R = 0.10 × 语法正确
  + 0.80 × 公共测试通过率
  + 0.10 × 公共测试全部通过
```

与 EXP-001 已实现的训练奖励口径一致。

## 计划的第一版 GRPO

初始目标：

- num_generations：4；
- beta：优先从 0 开始，先避免额外 reference model 显存；
- 温度：与当前采样基线一致；
- 单卡 Tang 4090 Laptop 16GB；
- 先小数据冒烟，再扩大到正式训练。

所有具体参数必须以 Tang 上固定 TRL 1.14.1 的**真实接口**为准，不按其它版本文档猜测。

## 环境问题记录

### 问题 1：GRPOTrainer 导入失败

首次读取 TRL 1.14.1 GRPO 接口时出现：

```text
ModuleNotFoundError: No module named 'requests'
```

原因：

TRL 的 GRPO trainer 导入 vLLM 客户端模块，而当前项目模型依赖没有显式包含 `requests`。

处理：

- 在 GitHub 项目依赖中增加固定 `requests==2.34.2`；
- 环境修复后再继续读取真实 GRPO API；
- 不在 Tang 临时修改源码。



### 问题 2：YAML 的 `no` 被解析为布尔值

第一次 16 题 / 2 步 GRPO 冒烟在创建 `GRPOConfig` 时失败：

```text
ValueError: False is not a valid SaveStrategy
```

原因：

PyYAML 将未加引号的：

```yaml
save_strategy: no
```

按 YAML 1.1 规则解释成布尔值 `False`。

处理：

```yaml
save_strategy: "no"
```

该问题发生在任何 rollout 或参数更新之前，因此没有产生 GRPO 训练结果。



### 问题 3：超时代码留下 Docker 容器

GRPO 冒烟完成后检查执行环境，发现历史评测中有若干 `python:3.11-slim` 容器持续运行。

根因：

- 执行器通过 `subprocess.run(..., timeout=...)` 启动 `docker run --rm`；
- Python 超时会结束 Docker CLI；
- 但候选代码所在容器可能继续运行；
- `--rm` 只有在容器本身退出后才生效。

修复：

1. 每次执行分配唯一 `coderl-lab-<id>` 容器名；
2. 添加 `coderl_lab=1` 标签；
3. 捕获 `TimeoutExpired` 后执行 `docker rm -f <name>`；
4. 增加清理脚本 `scripts/cleanup_executor_containers.sh`；
5. 新增单元测试验证超时清理命令一定发出。

正式 GRPO 扩大训练前，必须用真实死循环候选验证容器不会泄漏。

## 冒烟门槛

- [ ] GRPOConfig / GRPOTrainer 在固定环境中可导入；
- [ ] 明确奖励函数参数传递方式；
- [ ] 明确从已有 SFT adapter 继续训练的方法；
- [ ] 16/32 题小规模 rollout 可执行；
- [ ] 公共测试 reward 工作；
- [ ] hidden tests 未进入 reward；
- [ ] loss / reward / advantage 有限；
- [ ] 无 OOM；
- [ ] adapter 可保存和重新加载；
- [ ] 结果写回 GitHub。
