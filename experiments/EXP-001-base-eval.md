# EXP-001：基础模型能力评测

## 状态

**基础评测流水线：已跑通。**  
**16 样本冒烟基线：已完成。**  
**同种子复现检查：待完成。**  
**正式科研数据集：待确定。**

## 研究目的

在任何监督微调或强化学习之前，建立可信的基础模型能力基线，并验证整条：

```text
任务 → 多次模型采样 → 代码归一化 → 容器执行
→ 公共测试奖励 → 隐藏测试 → Pass@k
```

流水线。

## 固定配置

配置文件：`configs/base_eval.yaml`

模型：

`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`

运行代码提交：

`504a3ae0a4ce0a6861bd61f7b520d0c4d3074878`

采样：

- 每题 16 个候选；
- 温度 0.8；
- top-p 0.95；
- 最大新增长度 512；
- 基础随机种子 42。

## Tang 实测环境

2026-10-06：

- GPU：NVIDIA GeForce RTX 4090 Laptop GPU；
- 显存：16376 MiB；
- NVIDIA Driver：580.178.04；
- Python：3.10.12；
- PyTorch：2.10.0+cu128；
- PyTorch CUDA：12.8；
- Transformers：5.18.0；
- Accelerate：1.15.0；
- Datasets：5.1.0；
- TRL：1.14.1；
- PEFT：0.21.2；
- Pillow：12.3.0；
- Docker：29.1.3；
- Docker image：`python:3.11-slim`；
- Docker digest：`sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5`；
- 项目目录：`~/projects/CodeRL-Lab`。

重要修正：后续显存预算按 **16 GB Laptop 4090** 设计，不按桌面 4090 24 GB 假设。

## 验证结果

### 1. GitHub CI

PR #1 当前 CPU 单元测试工作流：

```text
17 passed
conclusion = success
```

### 2. 本地可信夹具

三个任务，每题 2 个手工可信候选：

- 每题 1 个正确、1 个错误；
- Pass@1 = 0.5；
- Pass@2 = 1.0。

### 3. Docker 隔离执行

修复临时目录权限后，同一批可信候选在 Docker 中得到完全一致结果：

- Pass@1 = 0.5；
- Pass@2 = 1.0。

因此本地可信执行与 Docker 执行的判定口径一致。

### 4. Qwen3-0.6B-Base 最小真实模型冒烟

每题 1 个候选、最大 128 新词元：

- 3 个任务；
- 2 个完全通过隐藏测试；
- Pass@1 = 0.6667。

失败样本 `is_palindrome` 只生成了自然语言，没有生成目标函数；因此失败属于真实模型输出失败，不是评测器误判。

### 5. 16 样本 EXP-001 冒烟基线

共 3 个任务，每题 16 个候选，共 48 个生成：

| 指标 | 数值 |
|---|---:|
| Pass@1 | 0.4167 |
| Pass@4 | 0.8297 |
| Pass@8 | 0.9655 |
| Pass@16 | 1.0000 |
| sum_squares 正确候选 | 11 / 16 |
| count_even 正确候选 | 6 / 16 |
| is_palindrome 正确候选 | 3 / 16 |
| 语法失败 | 19 / 48 |
| 公共测试全通过 | 20 / 48 |
| 隐藏测试全通过 | 20 / 48 |
| 公共全通过但隐藏失败 | 1 / 48 |

平均训练奖励：0.46875。  
平均公共测试通过率：0.45833。  
平均隐藏测试通过率：0.45833。

生成阶段耗时：71.74 秒。  
整条命令运行时长：约 99.75 秒。

PyTorch CUDA 统计：

- 峰值 allocated：1294329856 bytes，约 1.21 GiB；
- 峰值 reserved：1367343104 bytes，约 1.27 GiB。

运行中 `nvidia-smi` 观察到模型 Python 进程约 1594 MiB，整卡总占用约 2055 MiB，GPU 利用率约 64%。

精炼结果已写入：

`results/exp001-base/`

## 实现验证日志

### 问题 1：Docker 镜像下载被错误计入单测试超时

首次 Docker 冒烟时 Tang 没有预缓存镜像，自动拉镜像与代码执行共享单测试超时。

修复：

- 执行器不再隐式拉镜像；
- 缺镜像时快速失败；
- 运行脚本在评测前显式准备镜像。

### 问题 2：Rootless Docker 无权遍历临时目录

首次镜像准备完成后全部候选报：

`python: can't open file '/work/runner.py': [Errno 13] Permission denied`

修复：

- 临时目录设为 0755；
- runner 与 candidate 文件设为只读 0444；
- bind mount 保持只读；
- 继续禁用网络、丢弃 capabilities、限制进程、内存和 CPU。

修复后 Docker 判定与本地可信执行完全一致。

### 问题 3：系统旧版 Pillow 与新 Transformers 冲突

GPU 虚拟环境为了复用系统 CUDA PyTorch 使用 `--system-site-packages`。系统旧版 Pillow 缺少 `PIL.Image.Resampling`，导致 Transformers/PEFT 导入失败。

修复：

- `model` 可选依赖增加 `Pillow>=10.0`；
- 虚拟环境最终使用 Pillow 12.3.0；
- 保留系统 PyTorch 2.10.0+cu128。

修复后 Transformers、TRL、PEFT 和 CUDA 均正常加载。

## 当前结论边界

当前任务只有三个，而且都非常简单。

因此：

**这些 Pass@k 数字只能证明评测和执行流水线工作，不能用于比较模型，更不能用于声称强化学习扩展了能力边界。**

下一步必须先确定正式代码任务数据集，并明确：

- 训练集；
- 验证集；
- 隐藏测试；
- 任务难度分层；
- 训练数据污染风险；
- 正式 Pass@k 采样预算。

## 下一步门槛

- [x] GitHub CI 单元测试通过；
- [x] Tang 基础环境核验；
- [x] Docker 隔离执行可用；
- [x] 0.6B 基础模型可完成多样本生成；
- [ ] 同种子 Pass@k / 生成结果复现；
- [x] 结果和环境信息写回 GitHub；
- [ ] 正式数据集方案确定。
