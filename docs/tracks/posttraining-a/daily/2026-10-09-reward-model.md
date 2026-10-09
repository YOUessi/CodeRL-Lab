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
