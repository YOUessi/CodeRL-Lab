# EXP-001：基础模型能力评测

## 状态

**代码阶段：第一版已建立，正在真实环境验证。**  
**GPU 模型实验：正在准备运行环境。**

## 研究目的

在任何监督微调或强化学习之前，建立可信的基础模型能力基线，并验证整条：

```text
任务 → 多次模型采样 → 代码归一化 → 容器执行
→ 公共测试奖励 → 隐藏测试 → Pass@k
```

流水线。

## 核心问题

1. Qwen3-0.6B-Base 在当前最小代码任务上的原始 Pass@k 是多少？
2. 多次采样是否能显著暴露 Pass@1 看不到的潜在能力？
3. 公共测试和隐藏测试之间是否存在明显泛化差距？
4. 当前执行器、结果文件和实验记录是否足以支撑后续 SFT / GRPO？

## 固定配置

配置文件：`configs/base_eval.yaml`

第一轮冒烟模型：`Qwen/Qwen3-0.6B-Base`

采样：每题 16 个候选，温度 0.8，top-p 0.95，最大新增长度 512，基础随机种子 42。

评测：Pass@1、Pass@4、Pass@8、Pass@16。

## Tang 实测环境

2026-10-06 首次环境核验：

- 设备：Tang；
- GPU：NVIDIA GeForce RTX 4090 Laptop GPU；
- 显存：16376 MiB（约 16 GB）；
- NVIDIA Driver：580.178.04；
- Python：3.10.12；
- Docker：29.1.3；
- 系统 PyTorch：2.10.0+cu128；
- PyTorch CUDA：12.8；
- 项目目录：`~/projects/CodeRL-Lab`。

重要修正：后续显存预算必须按 **16 GB Laptop 4090** 设计，不按桌面 4090 24 GB 假设。

## 两阶段执行

### A. CPU 可信夹具测试

```bash
pip install -e ".[dev]"
pytest -q
bash scripts/run_smoke_eval.sh
```

2026-10-06 实测：

- 17 个单元测试全部通过；
- 可信样例共 3 个任务，每题 2 个候选；
- 每题恰好 1 个候选通过全部隐藏测试；
- Pass@1 = 0.5；
- Pass@2 = 1.0。

### B. Tang RTX 4090 基础模型实验

GPU 虚拟环境采用 `--system-site-packages` 复用 Tang 已安装的 CUDA PyTorch，避免重复下载另一套大型 PyTorch/CUDA 运行时。

```bash
python3 -m venv --system-site-packages .venv-gpu
source .venv-gpu/bin/activate
pip install -e ".[dev,model]"
bash scripts/run_base_eval.sh
```

模型生成代码必须走 Docker 执行器。

## 实现验证日志

### 问题 1：Docker 镜像下载被错误计入单测试超时

首次 Docker 冒烟时，Tang 尚未缓存 `python:3.11-slim`。原实现让 `docker run` 自动拉镜像，但每个测试只有 5 秒执行超时，导致镜像下载和代码执行共享同一超时预算。

修复：

- 执行器不再隐式下载镜像；
- 缺少镜像时立即给出明确错误；
- `run_base_eval.sh` 在评测前显式准备镜像。

首次下载镜像摘要：

`sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5`

### 问题 2：Rootless Docker 无权遍历临时目录

镜像准备完成后，容器能启动，但全部候选都报：

`python: can't open file '/work/runner.py': [Errno 13] Permission denied`

原因：Python `TemporaryDirectory` 默认权限为 0700，而 Tang 当前 Docker 环境下容器用户映射不能遍历宿主临时目录。

修复：

- 临时工作目录显式设为 0755；
- `runner.py` 和 `solution.py` 设为只读 0444；
- Docker bind mount 仍为只读；
- 容器仍保持无网络、只读根文件系统、丢弃 capabilities 和进程/内存限制。

修复后 Docker 冒烟重新得到与本地执行完全一致的结果：

- Pass@1 = 0.5；
- Pass@2 = 1.0；
- 三个任务均为 2 个候选中 1 个完全通过隐藏测试。

### 问题 3：系统旧版 Pillow 与新 Transformers 冲突

为复用系统 PyTorch，GPU 虚拟环境使用了 `--system-site-packages`。第一次安装模型依赖后，Transformers 在导入图像工具模块时访问系统旧版 Pillow，报错：

`AttributeError: module 'PIL.Image' has no attribute 'Resampling'`

随后 PEFT 因 Transformers 导入链失败而无法导入。

修复策略：

- 在 `model` 可选依赖中显式加入 `Pillow>=10.0`；
- 让虚拟环境内的新 Pillow 覆盖系统旧版 Pillow；
- 继续保留系统 CUDA PyTorch，避免重复安装大体积 GPU 运行时。

该修复待 Tang 重新安装依赖验证。

## 必须记录

正式模型运行后继续补充：

- Git Commit SHA；
- CUDA / PyTorch / Transformers 版本；
- 模型下载版本/修订；
- 总运行时长；
- 峰值显存；
- 每题正确候选数；
- Pass@1 / 4 / 8 / 16；
- 失败类型分布；
- 原始生成样例；
- 任何异常。

## 当前不做的结论

0.6B + 三个样例任务只用于验证系统，**不能据此声称强化学习是否扩展能力边界**。

后续必须更换为规模足够、训练/评测严格隔离的正式任务集，才进入科研结论阶段。

## 下一步门槛

- [ ] GitHub CI 单元测试通过；
- [x] Tang 基础环境核验；
- [x] Docker 隔离执行可用；
- [ ] 0.6B 基础模型可完成多样本生成；
- [ ] Pass@k 结果可复现；
- [x] 首轮环境与失败信息写回本文件；
- [ ] 正式数据集方案确定。
