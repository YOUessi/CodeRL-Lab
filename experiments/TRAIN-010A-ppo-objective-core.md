# TRAIN-010A：PPO/GAE 数学目标和训练约束的工程基线

**2026-10-10。状态：已完成可复现的 CPU 数学目标与单元测试；尚未实现 Actor-Critic 的真实 PPO GPU 训练器。**

## 为什么需要一个独立模块？

CodeRL-Lab 的 TRAIN-009A 已使用真实 Qwen3-1.7B UltraChat SFT Actor 和 TRAIN-008A 独立训练的标量 Reward Model，用 TRL 1.14.1 GRPO 完成2步真正策略梯度+SFT参考Adapter KL；实验代码/分支见 Draft PR #33。但这**不是PPO**：

1. GRPO组内奖励归一化估计Advantage，无训练的价值函数；
2. 当前项目锁定TRL1.14.1无`PPOTrainer`导出；
3. 要称PPO需独立 Value model、GAE、old policy snapshot、ratio-clipped surrogate、clipped value-loss、KL、entropy、终止状态和完整rollout收集/重算机制；
4. TRAIN-009A v2因四个completion全被mask掉导致`global_step=1`但`grad_norm=0`、新模型权重SHA没变；v3显式放行截断样本，确实2步梯度不为零，但**8/8回答均256-token截断，零自然完成回答**，不能凭RM奖励上升就声称真实质量改进。这揭示把terminal/截断语义明确写入PPO的重要性。

因此，新建独立`feat/train010a-ppo-objective-core`，基于稳定Reward Model Track A父分支，不依赖TRAIN-009A不完整回复策略的结果。先实现**不依赖模型的数值真值基线**，让后续torch实现每一个Tensor对应的语义有可测试依据。

## 已实现的算法公式

PPO数据形状：`[batch, completion_tokens]`，`valid_mask`只代表有效completion token，非prompt部分；`terminal_mask`的EOS必须是该序列最后一个有效token。超长未自然结束的样本属于`truncated`，与EOS完成不同。

### 1. GAE（广义优势估计）

[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)
]

[
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1},
\qquad R_t=A_t+V(s_t)
]

其中`d_t=1`表示自然EOS终止，终止后不bootstrap；截断样本仍按明确的最后状态`V(s_T)`bootstrap。padding位置的`adv/return`固定为0，不进入损失。

### 2. PPO ratio与剪切策略损失

[
\rho_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{\rm old}(a_t|s_t))
]
[
L_{\rm policy}=-\mathbb E_{t\in\mathrm{valid}}
[\min(\rho_t A_t,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)].
]

实现中显式检查`0<epsilon<1`、数值有限、log-ratio极端溢出与零有效token，不让一个空mask误报成功。

### 3. PPO裁剪Value损失

[
V^{\mathrm{clip}}=V_{\rm old}+\mathrm{clip}(V_\theta-V_{\rm old},-\epsilon_v,+\epsilon_v)
]

[
L_V=\tfrac12\mathbb E[\max((V_\theta-R)^2,(V^{\mathrm{clip}}-R)^2)].
]

同时暴露Value clip fraction，不把纯Actor训练当Actor-Critic PPO。

### 4. 冻结SFT参考KL惩罚与Entropy

相对冻结的SFT参考Token对数概率，采用`e^{logp_ref-logp_new}-(logp_ref-logp_new)-1`的非负近似KL估计；总损失：

[
L=L_{\mathrm{policy}}+c_V L_V+\beta D_{KL}-c_H H.
]

控制Value系数、KL系数、Entropy系数均非负，KL参考必须在策略更新前真实权重SHA/逐Tensor确认，并与`old_logp`策略快照区别开。

### 5. 对未结束响应的处理

`validate_termination()`要求每条样本的Valid Token连续、EOS至多一个且必须是最后token；默认**全部样本截断时拒绝宣称PPO有效**。显式`assign_terminal_reward()`仅在自然EOS时支付所学Reward Model分数；截断回复不直接获得高奖励，而获得固定惩罚。由于缺少下游独立真人偏好质量对照，该惩罚只能当工程先验，正式数值需要后续单独预注册。

## 工程证据和限制

- `src/coderl_lab/train/ppo_objectives.py`：使用NumPy的纯CPU参考实现，不与特定PyTorch/TRL版本绑定，作为未来Torch训练的数学对照Oracle。
- `tests/test_ppo_objectives.py`：测试终止/截断分类、返回与GAE手工结果、正负Adv剪切、value clipping、reference KL、极端exp和零梯度有效Token失败保护，GitHub Actions CPU CI已实际成功。
- 当前 **不从任何模型中采集Rollout，不加载Qwen或奖励模型，不包含PPO优化器、独立Value Head、checkpoint、更没有真实PPO性能结果**。严禁以此PR标题或README谎称PPO已完成。
- 仍需实现：PyTorch differentiable actor/value-forward；Reward Adapter加载与冻结、旧/参考策略固定；EOS/PAD/Reward终止性；GRPO之前暴露的`all truncated`问题在PPO上避免；分批rollout、GAE+Loss、optimizer+checkpoint、single-GPU及独立heldout对照与reward hacking诊断。

当前里程碑从“PPO概念”推进到**算法公式已由可执行测试约束**，不越过GPU验证边界。大型训练仍以GitHub代码、Tang GPU真机、按日期日志管理。
