# TRAIN-014B：提高GRPO奖励可区分度的算法题实验（与014 v1严格隔离）

日期：2026-10-15，v2生成训练前冻结。TRAIN-014 v1在Tang 4090已完成两步烟测和12步训练、真实Docker检查与Adapter保存，但36道简单整数函数任务被1.5B-Instruct的采样完全解决：12步涉及24条候选 **24/24 全通过**、每组奖励标准差0、全部梯度范数0，保存权重SHA与初始LoRA一致。因此v1是工程闭环成功，但**不存在真实梯度学习**，不得称作有效强化学习提升。

本v2明确为工程难度修订，重新生成与v1不重叠、明确输入输出的 `solve(x)` 算法任务，涵盖：
1. 整数数组两数和的不同索引对数；
2. 整数数组非空连续子数组和可被d整除的个数；
3. 相邻增量至少为阈值g的最长连续递增片段；
4. 数组中严格大于k的连续元素段数；
5. 最大非空连续子数组和减去常数k，空数组规定0；
6. 字符串无重复字符最长连续子串长度加偏置k。

每类固定6个训练参数、2个留出参数，**36个train、12个heldout**；每题冻结12个以上可执行输入输出测试，覆盖空/负数/相等/重复等边界，训练oracle只交给真实Docker奖励函数不进prompt，heldout oracle只放独立文件。任务与参数不跨split复用。样本源SHA先写GitHub后跑GPU，不允许根据后验效果挑选参数。

模型 `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，TRL1.14.1、BF16底座、LoRA(r8,alpha16,q_proj/v_proj)、FP32 trainable；GRPO设置 `num_generations=4`、`per_device_train_batch_size=4`、temperature0.8、top_p0.95、max_completion_tokens 192、LR1e-5、seed42、beta0、use_vllm=False。奖励为严格实际Docker验证的`0.1 + 0.9*(通过输入数/全部固定输入数)`（全正确=1），语法/solve入口无效=0。保留逐个reward、group variance、梯度范数与冻结日志；未知Docker基础设施错误中断而不是计为模型错。先2步烟测，若组内差异和梯度均非零才做16步正式GRPO；若没有则保留失败记录、不开虚假的训练成功结果。

即使真实梯度非零，基础算术/数组/字符串题也不能说明能改善复杂真实软件工程；必须同种子对照未经训练的基线、冻结12题 `pass@1/pass@4` 并记录置信区间，不能以训练loss或奖励代替泛化。所有代码配置/测试/日志在GitHub，GPU仅Tang，不覆盖TRAIN-014 v1。