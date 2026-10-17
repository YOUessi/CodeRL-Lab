# 2026-10-17｜EXP-023 三臂7B GRPO奖励机制消融：最终训练验收与既有外部分布对照

## 控制设计
所有实验均在 Tang RTX 4090 Laptop GPU 16GB 进行，使用完全相同的 Qwen2.5-7B-Instruct 模型revision、TRAIN-011 冻结 SFT adapter（SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`）、TRAIN-017 48条冻结训练任务（SHA256 `21a63fcef2e74d7e902480db9aabceb6b61d5f3bd4aff551e1959f7b8e38e4bb`）、seed42、32 optimizer steps、每组4生成、同一采样温度、学习率及LoRA配置。三个独立输出目录，未覆盖历史结果。

格式奖励：静态AST通过且具有顶层solve函数奖励1，否则0。执行代理奖励：在固定Docker下无顶层异常，包含wrong_answer也得1，否则0；原判题驱动吞掉某些逐输入异常，因此该组不是严格逐样本无异常执行奖励。完整奖励：原有Docker测试通过比例的0.1+0.9×fraction计算，无效代码为0。

## 三臂正式训练实测
| 指标 | 格式 | 执行代理 | 完整奖励 |
|---|---:|---:|---:|
| GPU优化步数 | 32 | 32 | 32 |
| 奖励调用数 | 128 | 128 | 128 |
| 组内变化奖励数（32个连续组） | 0 | 0 | 29 |
| 平均奖励（定义不同，不横比） | 1.000 | 1.000 | 0.718835453125 |
| 与SFT起点逐元素变化数量 | 0 | 0 | 1,809,257 |
| 训练后adapter SHA256 | 08d5ec82e1a1e421192a59ddb3f58cc9272315b5c78e00f1f62cad0c2f132f6f | 08d5ec82e1a1e421192a59ddb3f58cc9272315b5c78e00f1f62cad0c2f132f6f | 47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4 |

三臂训练状态均为 real_GPU_GRPO_completed，但前两臂**没有**有效参数更新，属于饱和奖励造成的零优势信号；只有完整奖励对参数产生真实变化。注意adapter文件SHA即使与源SFT文件SHA不同也可能是重新序列化导致，参数逐元素完全一致才是前两组“未学习”的关键依据。

## 精确历史复现与外部结果
本次完整奖励adapter SHA256与 TRAIN-017 旧 seed42 完全一致，reward histogram也一致；原始reward_attempts文件hash不同，因为本实验日志新增reward_arm等记录字段。由此可严格复用**同一适配器**的已冻结EXP-021种子42 MBPP+96结果，而不是声称此轮又重新生成了外部答案：

- 固定SFT：正确候选216/384，首样本正确56/96，4样本至少一次正确69/96。
- 完整奖励GRPO（同哈希seed42）：正确候选217/384，首样本正确56/96，4样本至少一次正确69/96；正确候选仅+1/384，即+0.2604个百分点；按题目配对bootstrap95%区间[-0.7813,+1.3021]个百分点，包含0。
- 同一批配对：文本改变60/384、错误→正确6、正确→错误5。真实存在局部修复及回退，但未证明总体显著提升。
- EXP-022随后对seed42局部修复案例进行数学输入追加检验，如 MBPPPlus/100（回文）与 /618（列表真除法），GRPO追加测试通过、相同冻结SFT未通过。其它种子的局部语义修复不应归功于这个seed42适配器。

由于格式与执行代理训练后模型参数逐元素与SFT完全相同，它们在同一个基础模型和推理配置下并未形成新的学习策略；不能把一个重新随机抽样产生的波动当作学习收益。

## 科研判断
本实验支持：**在本数据、单一训练seed和奖励定义下，只有带逐项测试正确率信息的完整奖励产生有效梯度更新，并出现有限局部语义修复及回退。**

不支持：所有格式/执行奖励都无效；奖励是语义修复的唯一因果来源；GRPO稳定提升了编程泛化能力。执行臂定义有代理混淆，需要升级为可严格判别真实执行成功的奖励；还需跨训练seed复现并加新的冻结外部数据域（HumanEval+ / 时间切分LiveCodeBench）。

## 复现路径
- 实验入口：`src/coderl_lab/train/exp023_7b_grpo.py`、`src/coderl_lab/train/exp023_reward_arms.py`
- 实验配置：`configs/exp023_7b_{format,execution,full}_seed42.yaml`
- 验收器：`src/coderl_lab/analysis/exp023_training_audit.py`
- Tang：`/home/you/projects/CodeRL-Lab-track-train017/artifacts/exp023/three_arm_training_audit.json`
- 独立奖励逻辑单元测试：3 passed。
