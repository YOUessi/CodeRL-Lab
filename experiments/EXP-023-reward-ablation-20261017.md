# EXP-023｜7B GRPO 奖励消融（2026-10-17）

三组从同一冻结 SFT adapter 起步，基础模型 Qwen2.5-7B-Instruct，NF4+BF16，LoRA float32，seed42，训练48题，holdout24，32 steps，4 generations/group，学习率与采样均保持 TRAIN-017 的冻结值。

奖励干预：
- format：静态 AST Python 合法且提供顶层 solve 函数则奖励1，否则0。无需运行候选代码。
- execution：使用已固定的 Docker 判题器，代码合法且无顶层异常则奖励1，否则0。**代理定义**：原有 DRIVER 会吞掉每个测试输入中的异常并将其归入 wrong_answer；因此此组不能称严格的“逐输入执行无异常”，需后续重做独立执行探针。
- full：此前 TRAIN-017 完整执行反馈：0.1+0.9×训练测试通过比例（无效代码为0）。

主要对照指标：训练是否真实更新、奖励组内方差、训练耗时、冻结 heldout24 和 MBPP+96 的候选正确率、Pass@1与四采样覆盖率、NameError与边界逻辑错误变化。禁止用训练奖励数值直接横向比较不同奖励尺度。

对照约束：三组使用相同训练 seed42 和 SFT adapter SHA256 25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8；各自独立输出目录和参数 hash；不覆盖 TRAIN-017/EXP-021。由于仅一个训练种子，目前只验证机制可行性，不能声称稳定效应。完整奖励对照可复用原有 seed42，但也准备独立执行配置。

当前环境：仅 Tang 执行（RTX 4090 Laptop 16GB）；代码在独立 `exp023_7b_grpo.py` 中，不改变历史训练器。
