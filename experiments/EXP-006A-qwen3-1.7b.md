# EXP-006A：Qwen3-1.7B 规模复现

## 状态

**设计：完成。**  
**Base smoke：已通过。**  
**SFT smoke：已通过。**  
**正式 Base / SFT validation：已完成。**  
**GRPO：配置已完成，等待 2-step GPU 冒烟。**

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


## Base 20题冒烟结果

代码提交：

`474621406977da3430fab89f62fc2b5a9a6ff00e`

设置：

- MBPP validation 前 20 题；
- 每题 4 个候选；
- 共 80 个 completions；
- 与 0.6B 相同生成参数。

结果：

| 指标 | Qwen3-1.7B-Base |
|---|---:|
| Pass@1 | **23.75%** |
| Pass@4 | **50.00%** |
| 语法失败 | 23 / 80 |
| 平均公共测试通过率 | 26.25% |
| 平均隐藏测试通过率 | 25.63% |
| 隐藏测试全通过候选 | 19 / 80 |
| solved tasks | 10 / 20 |
| 4/4 全正确任务 | 0 |
| 生成耗时 | 241.04 s |
| 峰值 allocated | 3,759,021,056 bytes |
| 峰值 reserved | 3,902,799,872 bytes |

同一 20 题冒烟中，0.6B Base 曾得到：

- Pass@1 = 6.25%；
- Pass@4 = 25.0%。

因此 1.7B 的基础代码能力明显更强，但正式结论仍以完整 90 题为准。

## 32样本 SFT 冒烟结果

设置：

- 32 个 train 样本；
- 1 epoch；
- micro batch = 1；
- gradient accumulation = 16；
- 有效 batch = 16；
- 其它 LoRA / 学习率参数与 0.6B 保持一致。

结果：

| 指标 | 数值 |
|---|---:|
| train loss | 1.04804 |
| Trainer runtime | 7.05 s |
| 可训练参数 | 17,432,576 |
| 总参数 | 1,738,007,552 |
| 可训练比例 | 1.003% |
| 峰值 allocated | 4,410,994,688 bytes |
| 峰值 reserved | 5,089,787,904 bytes |

结论：

> Tang 的 16GB RTX 4090 Laptop GPU 可以稳定运行 1.7B LoRA SFT，当前没有显存瓶颈。

因此 EXP-006A 可以直接进入 374 样本 × 3 epoch 正式 SFT。


## 正式 1.7B SFT 训练结果

训练代码提交：

`5fc65a5486de90da88679f6e3ee698f792c76a1a`

设置：

- 374 个 train 样本；
- 3 epoch；
- 72 optimizer steps；
- micro batch = 1；
- gradient accumulation = 16；
- 有效 batch = 16；
- learning rate = 2e-4；
- warmup steps = 4；
- LoRA r=16 / alpha=32；
- seed = 42。

结果：

| 指标 | 数值 |
|---|---:|
| train loss | **0.45758** |
| Trainer runtime | 180.96 s |
| samples/s | 6.20 |
| steps/s | 0.398 |
| 可训练参数 | 17,432,576 |
| 可训练比例 | 1.003% |
| peak allocated | 4,443,645,952 bytes |
| peak reserved | 7,432,306,688 bytes |

正式 SFT adapter SHA-256：

`e1ea9a007a3e4213cdeb1ca6b1e12d29adb83594991421757d0d40b9168d0861`

结论：

> 1.7B LoRA SFT 在 Tang 16GB 上稳定完成，峰值 reserved 约 7.43GB，因此后续 1.7B GRPO 有实际尝试空间。


## 1.7B GRPO 配置冻结

正式 SFT adapter 已固定：

`e1ea9a007a3e4213cdeb1ca6b1e12d29adb83594991421757d0d40b9168d0861`

GRPO 第一版完全复用 0.6B 的算法设置：

- num_generations = 4；
- max completion length = 256；
- per-device batch = 4；
- gradient accumulation = 2；
- learning rate = 1e-6；
- loss_type = grpo；
- beta = 0；
- 公共测试奖励；
- hidden tests 禁止进入训练；
- 4 路 reward worker。

先运行 16 题 / 2 step 冒烟；只有 16GB 显存可承受时才进入完整 187 step。


## 正式 1.7B Base validation

统一评测：

- MBPP validation 90 题；
- 每题 4 候选；
- 共 360 completions；
- 与 0.6B 使用相同温度、top-p、长度、代码归一化和 Docker 隐藏测试。

结果：

| 指标 | 1.7B Base |
|---|---:|
| Pass@1 | **33.61%** |
| Pass@4 | **64.44%** |
| 语法失败 | 92 / 360 |
| 平均公共测试通过率 | 34.72% |
| 平均隐藏测试通过率 | 34.72% |
| 隐藏测试全通过候选 | 121 / 360 |
| solved tasks | 58 / 90 |
| 4/4 全正确任务 | 4 |

生成 360 候选耗时：394.60 秒。  
峰值 reserved：3,913,285,632 bytes。

对比 0.6B Base：

| 指标 | 0.6B Base | 1.7B Base |
|---|---:|---:|
| Pass@1 | 13.61% | **33.61%** |
| Pass@4 | 38.89% | **64.44%** |
| solved tasks | 35 | **58** |

说明 1.7B Base 的代码能力底座显著更强，后续 SFT / GRPO 的边际收益需要重新测量，不能把 0.6B 结论直接外推。


## 正式 1.7B SFT validation

统一 90 题 × 4 候选评测：

| 指标 | Base | SFT | 变化 |
|---|---:|---:|---:|
| Pass@1 | 33.61% | **36.67%** | +3.06 个百分点 |
| Pass@4 | **64.44%** | 53.33% | **-11.11 个百分点** |
| 语法失败 | 92 / 360 | **1 / 360** | -91 |
| 平均隐藏测试通过率 | 34.72% | **38.61%** | +3.89 个百分点 |
| hidden-all-pass candidates | 121 | **132** | +11 |
| solved tasks | **58 / 90** | 48 / 90 | **-10** |
| 4/4 全正确任务 | 4 | **19** | +15 |

这说明 SFT 同时发生两件事：

1. 把一部分题的成功概率显著集中，导致 4/4 全对题从 4 增加到 19；
2. 但探索覆盖收缩，至少一次能答对的任务从 58 降到 48。

### 按题变化

- SFT 新从 0→至少1个正确候选：5 题；
- Base 原本可解、SFT 变成 4次全错：**15 题**；
- 正确候选数增加：29 题；
- 正确候选数减少：22 题；
- 不变：39 题。

### 配对 bootstrap（20,000次）

#### Pass@1 任务成功率差异：SFT - Base

- observed delta：+3.06 个百分点；
- 95% CI：[-3.89, +10.00] 个百分点；
- P(delta > 0)：0.7947。

当前不能说 Pass@1 提升在统计上可靠。

#### Pass@4 / solved-task coverage：SFT - Base

- observed delta：**-11.11 个百分点**；
- 95% CI：**[-20.00, -2.22]** 个百分点；
- P(delta > 0)：0.00645。

因此当前可以明确说：

> **在 Qwen3-1.7B-Base 上，这套 SFT 显著降低了 4-sample 任务覆盖率，同时大幅提高输出语法有效性和部分任务的成功概率集中度。**

这比 0.6B 上“Pass@1 增益大于 Pass@4”更强，说明概率集中/覆盖收缩并不是 0.6B 容量特例。
