# 2026-10-10：TRAIN-007D 相同偏好集 SFT/DPO 评测准备

## 需求

此前正式DPO已完成251步，并对256原始/252有效UltraFeedback heldout报告了 `eval_rewards/accuracies=54.365%`。那是TRL `β(logπ_policy−logπ_ref)` 参考相对奖励，**不是初始SFT以同一标尺独立评测的偏好准确率**，不能写成DPO优于SFT的增益。需要两份模型在同一252对、同一Tokenizer、同一1024-token规则下对chosen/rejected分别计算补全对数概率。

## 03:13—03:21：确实开发、测试与确认的事情

- 基于`feat/posttraining-track-a-data-qlora`独立创建 `exp/train007d-matched-sft-dpo-preference-eval`，并建立 Draft PR #31，不改母PR已有正式DPO结果。
- `src/coderl_lab/analysis/train007d_preference_eval.py`已实现全量 `256 raw→252 effective` 冻结过滤、同一NF4基础模型加载SFT+ DPO两个只读Adapter、单对回答logp与平均Token/总Token双口径偏好准确率、逐题只存数值的resumable log、2万次配对bootstrap/seed42。
- 正式两份权重：SFT Adapter SHA `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`，DPO Adapter SHA `3cb6ff9bd51e273b8c77bcf07f574a708b1ad2b089e53f9b61f4dc265c93ced4`，source UltraFeedback train/heldout SHA均与母分支冻结实验相同。
- GitHub [完整CPU Actions对照数据预检](https://github.com/YOUessi/CodeRL-Lab/actions/runs/37979244163)通过，真实Qwen tokenizer确认256原始heldout→252有效题（未按模型正确率筛题）。`tests/test_train007d_preference_eval.py` 5项CPU专项单测和GitHub CI通过。
- 在Tang的`/home/you/projects/CodeRL-Lab-track-eval`创建只用于真机GPU测试的独立detached工作树，绑定GitHub SHA `3a09e7e1c0379ffd7f12863074e07ca776c8d700`，真实读取并重哈希SFT/DPO权重、训练/偏好manifest/测试数据，全部验证通过。
- `scripts/run_train007d_matched_eval.sh`执行真实GPU单卡exclusive检查和同一252对对比；`scripts/queue_train007d_after_rm.sh`从03:21在Tang创建**单次本机条件队列**，正在等待TRAIN-008A正式Reward Model完整256步完成。前置失败、源代码漂移、他项目占GPU或RM未产生验证准确率时不得启动。
- 本队列仅为设备本机受控执行进程，**不是ChatGPT后台完成通知**，不保证结果已经产生。

## 当前可诚实报告的边界

- **03:21尚未实际运行TRAIN-007D GPU评分，尚无SFT/DPO paired accuracy新数值**。
- 本次是基于已经见过UltraFeedback heldout DPO结果之后补充的后验同口径诊断，不能作为新前瞻性无污染benchmark，也不能证明生成题目正确率或PPO收益。
- 原有EXP-004L/N跨分布负结果完全不涉及此次方法/超参选择。

## 03:50 自动串行评测失败，05:06 修复并重启独立配对实验

- 依照先前串行条件，TRAIN-008A正式256-step模型及256 heldout、adapter SHA于03:49结项；`TRAIN-007D`本机队列在03:50通过RM完成校验、GPU独占、两份SFT/DPO正式模型和数据版本SHA。
- 进入实际评测代码时因**路径类型不一致**失败：`verify_frozen_sources()`中的 `sha(sft_adapter)` 期望权重文件，但运行时传入PEFT加载所用Adapter目录 `.../train007a-qlora-ultrachat`，抛出 `IsADirectoryError`。**这次尚未计算任何配对log概率、没有 `task_logps.jsonl` 或 `summary.json`**，原失败日志保留。
- 在GitHub研究分支中明确新增 `adapter_weights_path()`：支持传入目录或直接 `adapter_model.safetensors` 文件时统一哈希唯一合法权重文件，缺失/非正规文件拒绝；没有改输入数据/超参/Tokenizer或训练权重。补2项回归测试。
- Tang在05:06拉取Github源 `e6cd2dbdea6813541ce64d63e511bd9499d876f6`，专项CPU 7项通过、真实两份Adapter weights及manifest/source SHA核对通过，RTX 4090仍空闲；原队列状态以明确失败原因标记，原日志未覆盖。
- **05:06:51** 在独立新路径 `artifacts/train007d-matched-preference-retry` 启动真实252-pair SFT/DPO GPU评测，重跑日志 `artifacts/train007d-matched-preference-retry-20261010.log`。该评测只做forward不训练；只有全部252对、两份adapter的逐题输出完整后才能报告配对accuracy/Bootstrap。
- 截止本日志初次写入，**新的GPU配对结果尚待实际生成**；不把失败队列文件改成成功，也不以单独DPO eval_reward_accuracy取代真正SFT基线。
