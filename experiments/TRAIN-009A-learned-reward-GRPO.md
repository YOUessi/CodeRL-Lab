# TRAIN-009A：首次用真实训练好的奖励模型做在线策略强化学习

**2026-10-10 首次预注册。当前代码路线为奖励模型驱动的 GRPO（Group Relative Policy Optimization），不是 PPO，也不属于已有的可验证代码测试奖励型 GRPO。**

## 从已验证结果出发

- TRAIN-007A UltraChat4096 训练的真实 Qwen3-1.7B QLoRA SFT LoRA：SHA `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`。
- TRAIN-008A UltraFeedback2048/256训练的真实 Qwen3-1.7B NF4 scalar reward head：SHA `586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c`。固定256 heldout在训练前后 44.92%→56.64%，但这只是固定偏好集证据，不自动代表任何RLHF策略增益。
- TRAIN-007C与TRAIN-007D表明DPO已经完成优化，却在同一252偏好验证集的 mean token logprob选择准确率上维持139/252（55.16%），是零翻转的固定样本事实。不能把 DPO参考相对奖励54.37%与 RM偏好排序56.64% 直接拿来排名。

## 工程路线选择

本项目的 `trl==1.14.1` 在 Tang 上经真实 `from trl import PPOTrainer` 兼容性检查失败（没有该导出），但现有GRPOTrainer及其 GRPOConfig可调用。为了先完成“冻结学习奖励→在线rollout→策略更新→KL控制→权重输出”的真实RLHF闭环，**先实现学习奖励驱动的GRPO**。PPO需要单独接入Value function / GAE / clipped surrogate /value loss，并独立验证训练器或实现，不把GRPO偷换成PPO。

## 训练集选择与奖励污染保护

Reward Model 已训练在 UltraFeedback train2048对，因此其偏好标签及 test256对 **不进入新的在线Policy数据集**。本次在 UltraChat SFT train4096中选prompt-only，并移除同任何 UltraFeedback train/heldout 以及 UltraChat SFT heldout256 精确prompt重复的记录。不把SFT原response传给RLHF trainer，只传prompt。对所有源文件、H4 manifests、prompt唯一性/交集再审计。

真正的 GitHub Actions 固定原始数据 [TRAIN-009A 真实数据审计](https://github.com/YOUessi/CodeRL-Lab/actions/runs/38020757383) 已执行：**4096条UltraChat train中实际4095条符合过滤条件、跨Reward两划分交集1条被移除**；从4095条按seed42 SHA固定选择Smoke16条。不含UltraFeedback heldout prompt，且不依据Reward或模型分数筛选训练prompt。未来正式独立验证应使用未进入在线策略训练和奖励模型训练的UltraChat validation prompts，但奖励分数本身不能代替人工真值。

## 算法及回报

- Actor：Qwen3-1.7B-Base NF4 Double Quant BF16 + **已经训练的 UltraChat QLoRA SFT Adapter**，而不是新随机LoRA。
- Reward：另一个 **冻结的训练完成的 NF4 RM Adapter+可训练时获得的score分类头**；在本RLHF阶段 `requires_grad=False`，不更新权重。
- Online rollout：同一prompt生成4个 completion，GRPO以该组回报中心化优势更新Actor；`learning_rate=1e-6`、BF16 NF4、GRPO `loss_type=grpo`、group reward scaling。
- Reward对生成的回答保留RM训练时冻结的表示方法：prompt最多512Token、回答空间、总长768Token+EOS。原始RM输出为标量，再用`tanh(r/2)`限定到[-1,1]，记录原始值的范围、奖励方差、饱和率、空答案数、组内重复率等，以监测简单奖励欺骗。
- 真正的KL：`beta=0.02`。TRl GRPOTrainer在PEFT已有SFTAdapter的情况下复制默认LoRA得到`ref` Adapter，并在训练阶段以其logprobs计算KL。首次optimizer step前**逐权重验证default/ref完全相同且ref全冻结**，确保不是不小心KL到裸Qwen Base。
- Smoke预算：16条固定prompt源、4generations/group、`max_completion_length=96`、`max_prompt_length=512`、1 optimizer step，且**单卡GPU互斥**。只验收正确发生了真实rollout、真实模型reward、策略梯度、KL与新Adapter持久化。
- 附加对照：零步的原始SFT同提示同采样基线、holdout固定prompts、before-vs-after同seed多样性和奖励/长度/KL/泛化的独立测试。这些还没跑，不预填正面结论。
- 本实验不使用任何 MBPP hidden tests、LiveCodeBench private tests，也不把之前两个benchmark上观察到的负结果用来调整奖励系数。
- **正式大预算训练没有确定或启动**。需先真实Smoke通过并核对log、冻结formal optimizer step budget、限定数据及负面对照后再启动；不因为这次是“强化学习”就默认自动大量跑GPU。

## 架构和工程证据

`src/coderl_lab/datasets/train009a_online_prompts.py`：四份H4冻结原始数据及两类manifest完整SHA和disjoint gate。

`src/coderl_lab/train/learned_reward_grpo.py`：真实冻结RM加载、NF4 SFT actor、GRPOTrainer、逐步验证SFT参考LoRA、group reward原始与bounded diagnostics、模型+日志保存。

`configs/train009a_learned_reward_grpo.yaml`：包含模型/data/Reward/算法超参固定版本，online与PPO命名边界。

`scripts/run_train009a_learned_reward_grpo.sh`：GPU锁、数据/模型SHA预检和1-step real smoke；如果已有结果或缺少权重则拒绝覆盖。

`tests/test_train009a_learned_reward_grpo.py`：合成多数据源隔离、模型SHA篡改、防NaN/空答案、缺KL禁止训练的单元测试。

前向模型大型checkpoint只在 Tang 允许目录；GitHub永久归档代码、配置、每日期工程日志和安全汇总+SHA。不要把未跑的策略效果写成已经实现的模型偏好改善。

## 当前下一步

1. CPU真实冻结数据审计已通过，并核对有且仅有1个跨UltraChat/RM prompt重合被过滤。
2. 在 Tang 的独立GPU验证工作树进行单卡1-step真实online RLHF smoke，先验证SFT ref Adapter副本与Reward真正冻结。
3. 若smoke真实存在优劣变化，保存运行值；如果发生资源不足、回报恒定、KL偏差或初始化错误，保留错误数据并修复。


## 2026-10-10 真实GPU执行后的结论与限制

- v1: 当前TRL1.14.1不接受 `GRPOConfig(max_prompt_length=...)`，模型加载后在训练器构造前失败；改为在模型生成前用真实tokenizer检查每条提示长度，禁止静默截断。
- v2: 1-step GRPO结构上执行完成，RM真实评分4个completion、ref adapter392组相等，但全部96-token截断且`mask_truncated=true`，**grad_norm=0、新Adapter SHA等于旧SFT SHA**。不能算任何策略更新，失败证据已冻结到 `results/train009a-learned-reward-grpo-smoke-v2/summary.json`。
- v3: 单独2-step GRPO工程smoke把max_completion改为256并**只在smoke显式解除truncated mask**，训练前保证`KL beta=0.02`与RM LoRA不更新。真机观察grad_norm分别0.55078和0.72656、KL 0.00145和0.00177、新 LoRA hash确实变化、Reward LoRA hash保持不变、无OOM。正式机器数据：`results/train009a-learned-reward-grpo-smoke-v3/summary.json`。
- **v3仍8/8回答达到长度上限，无自然结束**；只是优化了被截断回答的冻结RM分数，不能声称RLHF改善真实完整回答，更不能把它视为PPO。单seed、一张4090、smoke16提示、2 optimizer steps；正式训练预算和独立验证仍未冻结或运行。
- 下一步应作为新预注册实验分析 SFT 策略的EOS终止行为，在不靠已看过的奖励正负结果选prompt的条件下固定长短任务和max completion，验证自然终止比例、组间reward方差、零梯度/裁剪风险，之后再决定正式mask=true的online GRPO是否有统计或工程可行性。完整PPO另建Value/GAE/clip必要测试，不与GRPO混用名字。
