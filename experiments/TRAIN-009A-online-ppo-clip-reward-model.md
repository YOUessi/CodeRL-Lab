# TRAIN-009A：以已训练奖励模型驱动的真实在线 PPO-Clip / RLHF 小规模闭环

## 2026-10-10 固定假设，执行 GPU 结果之前登记

**任务定位：真实 RLHF（在线 rollout → 冻结 Reward Model → 参考SFT策略KL约束 → 可训练价值头/GAE → token-level PPO-Clip策略更新）的 CUDA 工程冒烟。不是先已有成绩再想办法宣传算法有效；不代表正式人类偏好或安全性提高。**

### 动机

此前主线A已完成4096/256 UltraChat QLoRA/BF16 SFT、2048/256 UltraFeedback DPO、独立NF4 reward model训练。奖励模型在固定256对heldout的偏好排序由44.92%到56.64%；DPO vs 原SFT在同252对、同一mean-logp准则下都为139/252，无偏好排序翻转。这些训练结果并不等于完成在线RLHF。

本次首次把**真实 SFT policy**与**真实已训练 Reward Model**连接起来，使用自己的学习器真正采样、估值、计算KL与反向更新，而不是静态DPO训练或把GRPO命名为PPO。

### 模型、数据与防泄漏

- 基础模型 `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`，NF4/double-quant，BF16。
- 策略和固定参考均从同一UltraChat训练出来的SFT LoRA Adapter，SHA `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`；策略为可训练`policy`，参考为**不可训练**`reference`副本。使用同一骨干避免额外base模型计算显存，两个Adapter参数不能互相覆盖；训练前必须测试策略/参考的初始token logprob差值小于0.02。
- 奖励模型使用已完成256步的独立Qwen3 sequence-classification NF4 LoRA，SHA `586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c`，**全程推断、禁止更新**；记录原始RM标量、终止响应、空响应和长回答截断事件。
- 仅从已冻结 UltraChat **train_sft 4096** 条提取prompt，不碰UltraChat/test_sft256及UltraFeedback/test_prefs256的标签/样本。公开SHA观察：UltraChat train 与 RM train有1个相同prompt，与RM heldout为0；代码明确剔除所有与RM train/两份heldout相交的prompt。
- 在剩余可用 train-only 提示中，对 `SHA256(seed42|PPO|promptSHA)`排序选择 **8个训练提示 + 4个只读探测提示**，全部在奖励/输出前冻结。每步2个训练prompt，每prompt采样2个completion，4步共16个episode。探测提示不参与梯度（这一版不主张已执行探测实验）。
- 从未访问MBPP hidden tests/LCB private tests，也不将 RM heldout 偏好标签用于更新。

### PPO数学目标（和DPO/GRPO区分）

对真实采样出的每个回答 Token `a_t`：
[
r_t=-\beta\left(\log\pi_{old}(a_t|s_t)-\log\pi_{ref}(a_t|s_t)\right)
+\mathbb{1}_{t=T}\,\operatorname{clip}(R_{RM}(x,y),-2,2).
]
其中 `β=0.02`，标量奖励仅加在回答最后一个Token上；参考模型永远是冻结SFT Adapter，不能偷换为未SFT的Base。

可训练价值头 `V_\phi(s_t)` 从每个采样状态的Qwen隐藏向量输出单标量；采用 `γ=1, GAE λ=0.95` 的逐Token TD优势，回报固定后运行PPO policy更新：

[
L_{clip}(\theta)=
-\operatorname{mean}_{t}\min\{\rho_t\hat A_t,
\operatorname{clip}(\rho_t,0.8,1.2)\hat A_t\},
\quad
\rho_t=\exp\big(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)\big).
]

总损失再加 `0.5 * MSE(value, GAE_target)`；每batch两轮策略优化，每个episode一小批次，初次参数明确不是GRPO的group-only baseline。正负优势的剪裁符号、GAE、KL terminal shaping、优势标准化都由`tests/test_train009a_ppo_math.py`回归检测。

### 单卡约束、输出和验收

- 预注册配置 `configs/train009a_ppo_clip_rlhf_smoke.yaml`。单个prompt最多保留128 token，最多采32回复Token；sampling temperature0.8、top-p0.95。策略LoRA LR=5e-6、价值头LR=1e-4、max grad norm1.0、PPO clip0.2、`4个online batches × 2 prompt × 2 completions × 2 PPO epochs`。
- `scripts/run_train009a_ppo_clip_smoke.sh`：先CPU验证每个输入SHA、SFT/RM训练摘要、冻结数据划分及8+4 sample hash，再取得独占4090的 `flock` 和 `gpu_preflight`。只在需要真实CUDA验证时从GitHub pull到Tang工作树；代码与结果摘要长期保存在Github。
- 每次rollout取真实旧policy采样的logprob和旧critics、同一Token的冻结SFT参考logprob、RM偏好分数（而不是人工编造reward）；随后GAE、clipped surrogate、critic MSE一起反传。记录每批reward、sampled KL、clip fraction、梯度范数与Token数量。
- 成功后保存**policy adapter safetensors、价值头 safetensors、optimizer/RNG状态、冻结输入ID与摘要**，核查SHA，禁止写入GitHub原始模型和训练提示文本。
- 终态必须显示真实4轮、16episode、32次小批次优化（不是256步正式RLHF）；参考adapter未被优化，且输入选择不存在标签泄漏。

### 研究边界与下一阶段

这是「将SFT、Reward Model与PPO串接起来」的第一版工程冒烟。RM本身只有单seed、同一固定UltraFeedback heldout上的56.64%排序准确率，存在奖励欺骗/训练域外泛化问题。不能把训练中`R_RM`上升宣称真实质量上升，也不能把小规模PPO训练等同大规模PPO研究成果。

后续若进行正式比较，应使用**尚未用于当前方法选择的独立盲测prompt/人工偏好集合**，另行注册无PPO参考策略、不同KL/奖励缩放、训练有无reward权重随机对照等条件；不得基于已见过的LCB v5/v6 private准确率/UltraFeedback heldout反向调本次超参。
