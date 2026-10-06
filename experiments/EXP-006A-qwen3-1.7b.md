# EXP-006A：Qwen3-1.7B 规模复现

## 状态

**设计：完成。**  
**Base smoke：待运行。**  
**SFT smoke：待运行。**  
**正式 Base / SFT：待运行。**  
**GRPO：等待正式 SFT adapter 哈希后再配置。**

## 研究问题

0.6B 已经观察到：

1. SFT 显著提高 Pass@1 和语法有效性；
2. SFT 对 Pass@1 的增益大于 Pass@4；
3. GRPO 相对 SFT 继续提高 Pass@1，但 Pass@4 没有同步提高；
4. 纯 GRPO 存在大量组内零方差 rollout；
5. 静态/在线采样都能提高有效更新率，但没有带来统计可靠的 validation 增益。

EXP-006A 不增加新算法，只问：

> 这些现象在更强的 1.7B Base 模型上是否仍成立？

## 固定模型

`Qwen/Qwen3-1.7B-Base`

固定 Hugging Face revision：

`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`

模型文件约 3.44GB BF16；Tang 为 RTX 4090 Laptop GPU 16GB。

## 固定数据

继续使用 MBPP-v1：

`google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c`

不改变 train / validation / hidden-test 口径。

## 第一阶段：Base

先在 validation 前 20 题 × 4 候选做 smoke：

- 验证模型 revision 可加载；
- 记录峰值显存和生成速度；
- 验证 Docker 评测；
- 检查输出格式。

通过后，正式 Base baseline 使用完整 90 题 × 4 候选。

## 第二阶段：SFT

LoRA 与 0.6B 尽量保持一致：

- r=16；
- alpha=32；
- dropout=0.05；
- q/k/v/o + gate/up/down；
- learning rate=2e-4；
- 3 epoch；
- 有效 batch=16；
- max length=1024；
- completion-only loss；
- seed=42。

为 16GB 显存安全，微批从 2 改为 1，梯度累积从 8 改为 16；**有效 batch 不变**。

先 32 样本 / 1 epoch smoke，再完整 374 样本训练。

## 第三阶段：GRPO

只有正式 SFT 完成后才创建配置。

必须：

1. 将 1.7B SFT adapter SHA-256 写入配置；
2. 先做 2-step smoke；
3. 记录实际显存；
4. 若 4 generations 在 16GB 可行，再跑完整 187 step；
5. 若 OOM，只调整内存相关变量，不改变 reward / 数据 / seed，再记录为工程差异。

## 核心比较

同一 validation：

```text
0.6B Base / SFT / GRPO
vs
1.7B Base / SFT / GRPO
```

重点不是绝对分数，而是：

- ΔSFT Pass@1 / Pass@4；
- ΔGRPO Pass@1 / Pass@4；
- zero-grad fraction；
- frac_reward_zero_std；
- hidden mean；
- solved-task coverage。

若 1.7B 仍出现“Pass@1 提升 > Pass@4 提升”以及高 zero-std，则说明现象更可能是训练机制问题，而不是 0.6B 容量特例。
