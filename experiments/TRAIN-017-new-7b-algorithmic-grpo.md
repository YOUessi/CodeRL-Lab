# TRAIN-017：7B GRPO 新算法题训练与保留集评测（冻结协议）

2026-10-15，在本实验任何 GPU 训练和模型答案/holdout 得分查看之前冻结。主模型：**Qwen2.5-7B-Instruct**，不是1.5B。此前 TRAIN-014/014B 1.5B 只作工程探索，TRAIN-016 7B 已完成16步 GRPO，在看过的12道合成题上 pass@1 8/12→10/12、pass@4 11/12→12/12；样本很小，不能作为独立泛化证据。

## 研究问题和比较组
- 用与旧TRAIN-014B完全不同的**六种新算法家族**训练7B SFT起点的GRPO，并固定新holdout检验训练前后正确率、覆盖率和跨种子稳定性。
- 六组算法：最长回文子序列删除判定、恰有k种字符的子串计数、指定gcd的子数组计数、平移裁剪后的柱状图最大矩形、跳跃上限约束的最少跳跃数、指定xor的子数组计数。不是旧数据中的简单数组计数/LIS/GREEDY题。
- 共48训练题（每家族8组参数）及24留出题（每家族4组不同参数），**每题20–40个确定性边界/随机测试**。后端给模型的仅为问题文本、函数签名 `solve(x)`；训练集答案测试仅可被Docker奖励回调读取，留出oracle与模型训练/生成文件物理分开。
- Python生成器 `src/coderl_lab/datasets/train017_new_algorithms.py`，冻结输出 `artifacts/train017/frozen24/`（Tang本地），只把来源SHA/任务个数写GitHub：
  - train SHA `21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb`
  - public heldout SHA `3f8768827ba8cba10d3555a58a3de5e69ec197a0192f1adb40ea05ae13c0e7a8`
  - private heldout oracle SHA `2cea9653952f7357a5f0ea549a32a94d36e3a0b46a9332d5b7d46ee84978a027`。
- 注意：这批题**第一次在CodeRL-Lab项目中出现**，评价之前不根据模型答题情况调参。但原算法概念本身不是新颖问题，不能保证Qwen预训练未接触过相似算法。训练与holdout按同算法家族不同参数分离，仍是分布内参数泛化，不能冒充真正跨领域基准。

## 必须以同一7B SFT作为训练起点
- 底座 `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`，NF4 double-quant，BF16计算。
- 原始7B TRAIN-011 QLoRA SFT Adapter SHA `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`，必须逐Tensor核实Trainer初始化时值与该SFT一致，保留原始权重不覆盖。
- TRAIN-016曾观察TRL量化后trainable adapter被转为BF16；TRAIN-017使用Trainer初始化后、optimizer创建前只将trainable LoRA改为FP32（真实dtype核查，2,523,136 trainable params），GRPO `beta=0`不复制整张7B独立参考模型，保持16GB显存可运行。
- 初始配置：GRPO 4生成/题，per_device_batch=4，max_new_tokens=256，LR=1e-5，温度0.8，top_p0.95，seed42，梯度检查点，2-step冒烟，正式32-step（若真实显存不足须冻结新的独立版本并保留失败记录）。
- 奖励规则为Docker真实运行 `r=0`无效语法/入口，其他 `0.1+0.9*passed/total`，全正确1.0。禁止对Python候选在Tang宿主执行。容器 `python:3.11-slim@sha256:d3846bce6e97d0ef5c32fc023300d5e39251ad791e6fdfaee41d2ba75246766e` 禁网、只读、非root、无Linux capabilities，给定执行时/内存/进程资源上限。未知容器基础设施失败不能悄悄算答错。

## 先冻结后评价：24题x三7B状态x4采样x2种子
- 3模型臂：
  1. `sft`：完整TRAIN-011 7B SFT。
  2. `previous_grpo`：原TRAIN-016 7B SFT→GRPO16（Adapter SHA `3d84428201f8802cdcde33cb64dd2c46ee4f2538de850cb2240b017ff2a237c8`）。
  3. `new_grpo`：TRAIN-017全32步（未训练前不得猜测其SHA）。
- 两个预先冻结的seed **42与314159**。每题每臂每seed采样四次。理论总量 **24×3×4×2=576份回答**。各臂使用同一Qwen聊天模板、max_new_tokens=256、temperature=0.8、top_p=0.95，逐`(task_id,seed,sample_id)`固定生成种子；不参考正确性选择种子。
- **只读public heldout prompt生成并哈希冻结576份源码后**，Docker评分器才允许打开 `heldout_oracle.jsonl`。在冻结前不得查看holdout正确率；若出于硬件/时间分阶段运行，仍使用同一冻结队列、不据第一阶段结果改变第二阶段。
- 评分 `pass@1`（固定四采样中的第一份）、`solved@4`（四份至少一份通过）、4样本中的正确比例，按题聚类配对Bootstrap (20k, seed42)，也分2个采样随机种子报告效果方向是否一致。注意24题与2随机种子不代表48道独立题，主CI按任务为单位做聚类。
- 对照解释以 **new_grpo vs sft** 为主，`previous_grpo`是探索性旧训练方法对照，不应据结果回调训练参数。成功必须指实际数据可证明效果，不能以非零梯度或较高训练reward直接宣称泛化改善。
- 所有工程源码/冻结参数/过程日志/错误修复与聚合结果按日期保留GitHub，模型、真实输出和标签只留Tang。其它不相关分支不合并。

## 预先定义的失败条件
GPU OOM、模型来源SHA不符、部分答案遗漏、group reward std=0的比例过高、混合list/string Arrow不兼容、Docker不可用、候选代码无法隔离、训练后Adapter未保存、heldout推理路径意外访问私有测试、评分器报告标签不一致等，都保留错误并停止错误结果宣称。修复单独提交且原数据身份不变，旧失败日志不能删。