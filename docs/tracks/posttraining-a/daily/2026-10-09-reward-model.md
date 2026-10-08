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
