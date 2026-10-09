# 2026-10-09：TRAIN-008A 冻结数据 Tokenizer 质量检查

## 本日真实工作

- 基于 `HuggingFaceH4/ultrafeedback_binarized@daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2` 前一日已冻结的真实2048 train / 256 heldout对，新增 `src/coderl_lab/analysis/reward_model_token_audit.py`。
- 确定严重潜在问题：将 `prompt+completion` 直接用 `truncation=True, max_length=768` 右截断，若长prompt超限，答案本身可能被删除，造成chosen/rejected两臂只剩同一提示内容、无有效偏好梯度。
- 训练前修复源代码 `reward_model.pack_reward_tokens`：共享prompt尾部最多512 tokens、强制至少128个回答Token预算，长completion保留后部并追加EOS；显式冻结配置 `max_length=768, max_prompt_tokens=512, min_response_tokens=128`。
- 新增 `tests/test_reward_model_token_audit.py`、原 `tests/test_posttraining_reward_model.py` 的长prompt、长答案保留和错误预算拒绝测试。
- GitHub Actions `37822736841` 真正下载冻结H4快照，使用 `Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1` 的**真实Tokenizer**，两套数据全量token化，无GPU/optimizer step，也不读任何private tests。
- 结果训练2048：prompt截断158、chosen截断240、rejected截断188；候选完全编码相同0对。
- 结果验证256：prompt截断11、chosen截断26、rejected截断21；候选完全编码相同0对。
- 测试和workflow均已成功。源日志及Workflow Artifact `11570525681`，机器可读常驻Git：`results/train008a-token-audit/summary.json`。
- 其他主线 GPU 仍在执行EXP-004N v5 全量167题SFT模型生成；Reward Model **尚未进行GPU训练**，不能报告reward ranking gain/PPO收益。

## 后续正式实验边界

- 先在空闲GPU上做TRAIN-008A 32对训练/16验证真实smoke，核验NF4+序列标量头梯度与参数保存；
- 全量2048/256训练必须在 DPO/B 试验不占GPU后单独执行。结果报告训练前后在同一冻结heldout偏好准确率及loss/mean margin；
- 这些都是偏好建模，不等于完整RLHF/PPO，尚需后续独立训练和验证。


## 17:52—17:53：隔离依赖修复与受控真机测试准备

- 为避免复现DPO遇到的bitsandbytes导入问题，`run_posttrain_a_reward_model.sh`支持 `QLORA_EXTRA_PYTHONPATH` 指向单独安装的bitsandbytes overlay，并在冻结完整2048/256偏好数据SHA、GPU占用检查之后、模型加载之前验证`torch.cuda.is_available()`和真实 `bitsandbytes`版本；不改全局/共享Python环境。
- GitHub新增 `scripts/queue_train008a_after_dpo.sh`，严格等待已确认的DPO parent PID，且必须验证正式DPO新的Adapter/训练数据SHA/256heldout eval已完成且数值有限，才允许奖励模型smoke（32训练/16验证）调用4090。若验证失败则拒绝，绝不标记成成功。
- Tang `/home/you/projects/CodeRL-Lab-track-rm` 是基于GitHub `81bf0a8994f171cbeb0ba7922feea97f9d472a44` 的 detached真机测试工作树，项目`data/generated/posttrain-h4-v1`通过符号链接读取先前 A 的冻结数据；16项CPU专项测试与真实数据2048/256 SHA通过。
- 本机队列从17:53启动，等待DPO parent PID `843075`，状态 `waiting_for_frozen_TRAIN007C_result`。**仍未开始任何RM优化步骤**；不预填偏好准确率或完成时间。代码/配置/过程记录始终以GitHub为唯一事实源，真实权重留Tang。


## 18:11：TRAIN-008A 真GPU烟雾完成，23:56验证入库

- 在TRAIN-007C于18:10完成正式Adapter与heldout核验后，Tang用户授权的一次性奖励模型队列通过校验，实际启动 `TRAIN-008A` NF4/LoRA标量排序头（Qwen3-1.7B-Base）真实4步GPU训练，CUDA模型保存、训练前后同一heldout对评测均完整成功。
- 源GitHub代码 `81bf0a8994f171cbeb0ba7922feea97f9d472a44`，训练32对偏好、独立验证16对，4/4步骤；mean training loss `0.70082979`、peak reserved显存3,015,704,576字节、wall≈39.29秒。
- 训练前heldout偏好准确率 `10/16=62.5%`，训练后仍 `10/16=62.5%`；pairwise loss `0.72133695→0.70697509`，平均margin `0.08202758→0.10349453`。损失/边际值改善，但**16个样本不足以主张泛化和真实效果提升，尤其准确率无改变**。
- 新奖励头Adapter SHA256 `eae9900e4e5781d272cc70931138dc8d5e07b5844e8dbd797ddfc1b0ec3f512e`，通过从实际权重重新计算哈希核对一致。
- 本次结束标记是 `smoke_only`，**完整2048/256 Reward Model训练并未启动**；尚无PPO/RLHF策略更新或强随机基线对比。
- 已把真实实验摘要归档 `results/train008a-reward-model-smoke/summary.json`，权重仍在Tang本机。原实验日志、tokenizer审计和训练记录全部保留。
- 23:56再次核查 Tang GPU 空闲，无CodeRL训练。正式大样本奖励模型和比较协议为后续独立验收内容，不伪报。
