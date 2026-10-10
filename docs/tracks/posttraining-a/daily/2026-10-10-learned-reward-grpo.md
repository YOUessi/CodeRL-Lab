# 2026-10-10：TRAIN-009A 真实奖励模型驱动在线强化学习（GRPO）排障记录

## 11:19 开始的研究/工程目的

Track A已真实完成：TRAIN-007A UltraChat4096 NF4 SFT；TRAIN-007C UltraFeedback2048/256 DPO；TRAIN-008A UltraFeedback2048/256 pairwise奖励模型；TRAIN-007D在同252对 heldout上SFT-vs-DPO logprob偏好139/252→139/252。奖励模型256heldout训练前44.92%→训练后56.64%，并不证明PPO策略训练有效。

用户请求继续：本次不重复SFT/DPO/奖励模型训练，而是补上独立Reward Model驱动的真实online rollout+RL策略梯度阶段。已验证本机 `trl==1.14.1` **没有导出 `PPOTrainer`**，因此先新增`TRAIN-009A`为 **GRPO**；不能标作完成PPO。PPO的value head / GAE / clipped PPO另立任务。

## 11:28—11:32 GH真实数据与单卡前置

- Github基于`feat/posttraining-reward-model`建独立 `feat/train009a-learned-reward-grpo`（待审阅）；所有代码与配置直接先写GitHub。
- 本项目GitHub Actions `38020757383`真实下载 `37725387735` 冻结的全部 H4 源，核对4096 UltraChat SFT训练/256 heldout与2048 UltraFeedback RM训练/256 heldout各SHA和stage/seed/源版本。
- UltraChat4096 train真实与UltraFeedback RM train+heldout合计只有 **1个相同prompt**；此prompt排除，剩余**4095条**online eligible。按seed42 SHA从中先取16条做Smoke；**SFT答案不送给Online trainer**，不使用任何Reward模型见过的提示，不读MBPP/LCB hidden测试。
- Train actor绑定真正TRAIN-007A LoRA SHA `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`；真Reward Model绑定TRAIN-008A NF4 LoRA+score权重SHA `586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c`。
- 开发方案用GRPO 4 completions/prompt，RM score经固定`pack_reward_tokens`保留回答+EOS，再`tanh(raw/2)`作边界防reward hacking，KL `beta=0.02`。TRL本机源码确认：如果模型已是Peft SFT adapter且beta非零，会复制初始SFT权重给`ref` Adapter作为KL reference；在首步前增加真实逐参数equals/frozen检查。
- Tang真GPU测试前16项？本次新分支实际7项、随后9项、最新10项专门CPU tests全部通过；单卡preflight和bnb 0.50.2通过。工作树`/home/you/projects/CodeRL-Lab-track-grpo-rm`只用于实际GPU试验。

## 11:32：Smoke v1接口失败（完整保留）

- 真模型加载成功，但运行至`GRPOConfig.__init__()`报`unexpected keyword argument 'max_prompt_length'`。固定TRL1.14.1无此kwarg；此时**没有Optimizer step或新policy adapter**。原源提交`f609c2b88c45a93dcbee281ff9757261486b8448`。旧输出 `train009a-reward-grpo-smoke/training_identity.json`及原日志保留。
- GitHub改为在进入GRPO前用**实际Qwen3 Tokenizer验证每条prompt长度 <= 配置Token cap**；超过则明确拒绝，不偷偷右截断。真实16条smoke prompt最大132 tokens，全部 <=512。GRPOConfig移除无效kwarg，并加边界回归测试。

## 11:35—11:36：Smoke v2形式完成，但没有真实RL策略更新

- 独立 `train009a-reward-grpo-smoke-v2/` 首次完整产生了真实RM奖励callback：4条不同生成回复，无空回复，bounded reward mean约-0.30797、std约0.40405、原始RM score range [-3.4375,0.2461]。
- KL参考Adapter已**真实逐参数校验392组default/ref权重完全相同，ref无可训练权重**，并非错误地把裸Base当原始SFT reference。
- 但是，4条completion全部在96 tokens长度上限被截断，因 `mask_truncated_completions=true`，TRL将这些样本全mask，`grad_norm=0`、`kl=0`、新policy Adapter SHA与输入SFT SHA**完全一致**。**此轮不能算真实RL更新成功**，即使TRL返回 `global_step=1` 与0 training loss。问题被保存为完整负面工程事实，禁止按成功发布。
- GitHub因此强化成功验收：`train009a`输出必须有至少一个真实非零梯度值、新LoRA权重SHA不同于初始SFT、Reward Model原权重SHA保持不变，否则明确失败，不写正式`run_summary.json`。

## 11:39：独立Smoke v3（运行状态待核验）

- 不覆盖v1/v2；新增`configs/train009a_learned_reward_grpo_smoke_v3.yaml`、独立`scripts/run_train009a_learned_reward_grpo_smoke_v3.sh`以及回归测试。
- 目的**只是验证实际在线策略梯度通路**：2步、每组4rollouts、max completion256/max prompt480、`gradient_smoke_allow_truncated=true`并设置`mask_truncated_completions=false`。该非屏蔽策略**严格Smoke-only**，正式RLHF仍默认mask truncation防止优化不完整答案。
- SFT actor/RM模型/data/β=0.02仍全部使用相同冻结哈希，不事后按回答正确与否挑样本或调threshold。变更仅为检验“梯度是否能真实通向policy”；即使Smoke更新成功也不能据此宣布质量提升。
- 真机Source Git SHA `737c247850616189576273de5884b48756a6f46e` 已拉取、10项专项CPU通过、GitHub Actions真实数据审计通过、GPU独占可用。11:39 Tang启动v3真实2步CUDA。**本条记录时尚未有v3最终梯度/新Adapter结果**。

## 科学约束

当前所有实验仍是一个单seed小模型，reward head只在UltraFeedback固定偏好数据表现出改善，在线rollout的RM reward与人的偏好/实答质量不能直接等同。后续正式RLHF应先独立固定rollout budget、completion质量约束、SFT on-policy基线、RM奖励黑客化审计、新heldout及KL阈值；不把“能生成奖励”和“能提高真实任务准确率”混为一谈。新的PPO需显式value head/GAE/clipped policy/value objectives，不能从现有GRPO冒称PPO。



## 11:40—11:42：Smoke v3 真实2步策略梯度成功，但所有回答依然未自然结束

- Tang冻结 `TRAIN-009A` v3 Source Git SHA `737c247850616189576273de5884b48756a6f46e`；配置SHA `82b87c17b17ccec426515fd280f69eb2a13218f0ff1cd7f3a44bd0e5a4498fb0`，原SFT Adapter SHA、Reward Model Adapter SHA与全部四个 H4 原始文件SHA、4095有效prompt/16-smoke的选择SHA都未改。
- 实际 GRPOTrainer `2/2` optimizer steps、2次**真实冻结Reward Model**计算callback，共8条生成，no empty, no duplicates；group reward mean std约0.4425，平均bounded reward约-0.12686，原RM模型在训练前后权重SHA完全一致。
- **实际梯度不为零：第1步grad_norm=0.55078125、第2步0.7265625**；reference KL分别 `0.00145319186`、`0.00176777714`，train_loss约`3.22e-5`。训练后LoRA真实新SHA `4869d985ed30562312751f38ed5883f46bf7671c0847587253930e66ad965cb3` **不同于源SFT SHA**，新Adapter已真实持久化。
- 训练器复制的**392组原SFT/reference LoRA参数在首步前经torch.equal逐组一致**、reference无trainable参数；非“假KL到裸Base模型”。
- GPU peak reserved=6,125,780,992 bytes，训练墙钟约69.03秒，执行完成后CUDA模型进程退出；资源无异常。
- **重大限制：v3 8/8 completion都达到256 Token cap，EOS自然终止0条**。因只为证明梯度通路，单独Smoke v3取消了mask_truncated（而正式默认仍mask=true）；故不得从本次reward、KL或LoRA变更声称完整响应质量提升、有效策略泛化或已完成PPO。
- 机器可读训练结果：`results/train009a-learned-reward-grpo-smoke-v3/summary.json`；v2零梯度负面结果：`results/train009a-learned-reward-grpo-smoke-v2/summary.json`；完整prompt交叉过滤：`results/train009a-prompt-audit/summary.json`。
- 结论：现已真正完成从SFT policy + **另行训练的RM** → online sample → RM评分 → GRPO grad step/KL → 新策略Adapter落盘的**工程级闭环**。但要进入具有质量保证的formal RLHF，下一块应先**修复终止/长度行为**并运行独立heldout/负面对照，不能把v3截断策略延伸到formal。
