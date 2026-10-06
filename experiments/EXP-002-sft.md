# EXP-002：监督微调基线

## 状态

**代码实现：进行中。**  
**32 样本 LoRA 冒烟：待运行。**  
**374 样本正式 SFT：待运行。**

## 研究目的

建立与基础模型完全可比的监督微调基线，为后续 GRPO 对照提供参照。

核心问题不是“训练损失能不能下降”，而是：

1. SFT 对 MBPP validation 的 Pass@1 / Pass@4 提升多少？
2. SFT 是否只是提高 Pass@1，而 Pass@4 变化较小？
3. 公共测试和隐藏测试之间的差距如何变化？
4. SFT 是否增加格式正确率 / 降低语法失败？
5. 后续在相同数据与评测条件下，GRPO 相比 SFT 多带来什么？

## 固定数据

MBPP-v1：

`google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c`

训练集：374。  
验证集：90。  
测试集：500。

SFT 参考实现只来自 train split。

## 第一阶段模型

`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`

作用：验证完整训练和评测闭环，不承担最终科研结论。

## 固定软件环境

沿用 EXP-001 已验证环境，不允许实验过程中自动漂移：

- PyTorch：2.10.0
- Transformers：5.18.0
- Accelerate：1.15.0
- Datasets：5.1.0
- TRL：1.14.1
- PEFT：0.21.2
- Pillow：12.3.0

2026-10-06 首次准备 EXP-002 环境时发现未锁版本的 `pip install -e ".[model]"` 会尝试升级到 PyTorch 2.14.1 和 CUDA 13 依赖。安装被立即中止，项目依赖随后改为上述固定版本，避免 EXP-001 与 EXP-002 环境不一致。

## LoRA

- r = 16
- alpha = 32
- dropout = 0.05
- 目标层：q/k/v/o + gate/up/down

训练损失仅计算正确代码补全部分，不计算提示词部分。

## 正式训练配置

- epoch：3
- batch size：2
- gradient accumulation：8
- 有效 batch：16
- learning rate：2e-4
- scheduler：cosine
- warmup ratio：0.05
- max length：1024
- gradient checkpointing：开启
- seed：42

## 冒烟门槛

32 个训练样本、1 epoch：

- [ ] 能正常加载固定模型 revision；
- [ ] LoRA 目标层匹配；
- [ ] loss 有限且可反向传播；
- [ ] adapter 能保存；
- [ ] adapter 能重新加载推理；
- [ ] Tang 16GB 不溢出；
- [ ] 训练结果写回 GitHub。

## 正式评测

在 MBPP validation 90 题上，基础模型和 SFT 模型使用完全相同：

- 生成提示；
- 采样温度；
- top-p；
- 最大输出长度；
- 每题采样数；
- 代码归一化；
- Docker 执行器；
- Pass@k 计算。

第一版使用每题 4 个候选，报告：

- Pass@1；
- Pass@4；
- 语法失败率；
- 公共测试通过率；
- 隐藏测试通过率。

后续能力边界阶段再扩大 k。
