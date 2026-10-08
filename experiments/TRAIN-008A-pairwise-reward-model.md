# TRAIN-008A：利用真实偏好数据训练独立奖励模型（Pairwise Reward Model）

## 状态：2026-10-08 预先冻结配置；实际 GPU 训练尚未执行

这是 CodeRL-Lab 原始主线 A 的 RLHF 基础环节。任务是自己训练一个能够输出**标量奖励**的模型，而不是只使用 public tests 作为 GRPO 回报。该奖励模型之后可以服务于 PPO/RLHF 或离线 reward diagnostics，但 **TRAIN-008A 本身并不是 PPO/GRPO/RLHF 训练的证明**。

所有开发在基于 Track A 的独立 GitHub 分支 `feat/posttraining-reward-model` 进行。当前 Tang GPU 被 TRAIN-007B BF16 训练占用，待资源可用后再运行真实训练，不并发。

## 研究问题及可检验假设

问题：同一 prompt 下，经过 UltraFeedback 的选择/拒绝偏好对训练后，模型能否在未参与训练的256对数据上，更高概率给 chosen 相对于 rejected 更高分？

- H0：训练后验证偏好准确率相对随机初始化分类头没有提高。
- H1：训练后验证偏好准确率提高且成对排序损失降低。
- 无条件报告初始准确率、最终准确率、margin、损失、训练曲线、权重 SHA、GPU 资源和失败原因；若不提升，保留负结果。
- **解释边界**：这种验证只能支持偏好标签拟合/迁移，不能证明回答真实性、稳定的人类偏好质量或下游 PPO 收益。

## 数据固定

- 源：`HuggingFaceH4/ultrafeedback_binarized`。
- revision：`daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2`。
- 从 Track A 已验证的 full-scan manifest 使用 `train_prefs` **2048对** 和 `test_prefs` **256对**，seed42；训练和验证 prompt 不交叉。
- train JSONL SHA-256：`945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77`；
- heldout JSONL SHA-256：`9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450`。
- 每对样本包含共同 prompt、chosen 和 rejected。两条候选均保留相同 prompt prefix，改变的只有回答内容。
- 不把 MBPP test / LCB private tests 加入训练。

## 模型与训练协议

- Qwen3-1.7B-Base，revision：`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`。
- `AutoModelForSequenceClassification(num_labels=1)`，新增可训练 `score` 标量头。
- NF4 四位量化/双重量化，BF16 compute；PEFT LoRA rank16/alpha32/dropout0.05，attention+MLP 七投影；`score` 头经 `modules_to_save` 保存。
- Token 上限768（统一右侧截断），1 epoch，单对前向，gradient accumulation8、AdamW、learning rate=1e-5、cosine scheduler、warmup3%、seed42。
- 预计数学上的更新预算为 **2048/8=256 optimizer steps**（不是声称已实际跑完）。
- 每32步保存含adapter、classifier head、AdamW optimizer、scheduler及CPU/CUDA RNG的检查点；最多保留4份。暂停恢复必须匹配config SHA、源数据 SHA和训练进度，不存在有效checkpoint则不能冒充续训。

训练目标为 Bradley–Terry 成对排序损失：

\[
\mathcal{L}=\log(1+\exp(-(r_\theta(x,y^+)-r_\theta(x,y^-))))+\lambda \frac{r_+^2+r_-^2}{2},
\quad\lambda=0.001.
\]

其中 `r_theta` 是序列分类头输出的一个实数，不是生成文本的概率；同样的 prompt 下 chosen 希望比 rejected 得分更高。损失的 L2 项约束奖励幅度，避免无约束奖励放大。

## 评测与门禁

- 首次 GPU 梯度更新**之前**在全部256对 heldout 上计算初始标量头的偏好准确率、ties、均值margin、成对排序损失，写入本地 `initial_heldout_metrics.json`。
- 完整训练后对同一固定256对做最终验证；报告准确率变化以及同一指标的loss变化。
- 新模型 checkpoint SHA和GPU峰值显存、训练耗时写入 `run_summary.json`。
- 首先运行真实32对训练/16对验证的 smoke。Smoke不算正式泛化结果；正式2048/256单独输出。
- 未完成训练之前不允许把任何结果写成“RLHF 已完成”或“模型更安全/更正确”。
- 与 B 的 EXP-004N 的 GPU 生成严格串行，不竞争单张4090。

## 工程入口

`src/coderl_lab/train/reward_model.py`、`configs/posttrain_a_reward_model_qwen3_1.7b.yaml`、`scripts/run_posttrain_a_reward_model.sh`。使用 `MODE=smoke` 或 `MODE=formal`，脚本先检查全量冻结数据和GPU独占。`tests/test_posttraining_reward_model.py` 覆盖数据质量、分区泄漏、loss符号、数值稳定性、检查点身份一致性。

## 后续条件

只有在奖励模型完整训练及 heldout 检查通过后，才考虑 PPO/RLHF；届时必须另立实验并实现在线rollout、固定参考策略、KL约束、有效奖励验证与Reward Hacking审计。当前不将 PPO 理论写成已实现。


## 2026-10-08 22:02：正式来源数据的 CPU 实盘审计已通过

- [GitHub Actions 真实数据审计](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37789195866) 使用 Actions token 下载以前的冻结源数据快照（不是单元测试编造的JSONL）。
- 完整2048/256条偏好数据、数据SHA、source revision、prompt train/heldout无交集、chosen/rejected共享同一提示而答案不同等检查均通过。模型 GPU 未训练。
- 机器可读归档：`results/train008a-preflight/summary.json`，独立保存预先冻结的两个选样文件SHA和工作流版本。
- workflow只把去标识化的小型审计摘要上传为Artifact，没有向Git上传训练/验证对话文本。
- 下一项验收仍是单卡 32/16真实Reward Model smoke，必须在007B和004N不再占用GPU时进行；正式偏好胜率/训练前后排序对比尚未产生。
