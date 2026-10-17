# EXP-024｜严格可执行性—依赖完整性—语义正确性奖励（2026-10-17）

## 冻结实验
- 设备：**Tang RTX 4090 16GB**；GitHub 为代码和实验摘要唯一事实源。
- 同一 Qwen2.5-7B-Instruct 模型 revision、TRAIN-011 SFT adapter SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`。
- 延续 TRAIN-017 的48条训练任务、32个GRPO步骤、4 generations、LoRA及采样参数；用训练 seed 43/44 做两次独立训练，不覆盖 EXP-023。
- 新增第三个 MBPP+ 冻结96题集合；与历史24+96题均无交叉，原始 Parquet SHA256 `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`。公开题目 SHA256 `6438a8abdfc5ec19edd98f717e03d17b2c8a13f5fbe25f07a63db7ee030755f7`，隐藏测例 SHA256 `e945ac3b28fbecc306e24468b796ec883758d10588c5ab73fd37362518259060`。
- 这是一批**未见过的新MBPP+题**，不是跨数据域（HumanEval+/LiveCodeBench）验证，不能误写为跨基准泛化。

## 新奖励（训练数据上的固定测试）
生成Python代码经AST和solve入口检查；候选代码**仅在固定、无网、非root、只读Docker中执行**，对每个训练测试输入分别捕获执行异常：
- `E`: 输入级成功执行比例（无抛错）。
- `D`: 不发生 `NameError` 的输入比例；顶层加载遇到NameError则整体归零。
- `S`: 输入输出语义正确比例。
- 合成 `R = 0.10 E + 0.10 D + 0.80 S`；代码不合法或加载异常为0。
- 相比 EXP-023 的粗粒度执行代理，这次逐测试执行且不会把 `wrong_answer` 一律奖到1.0。
- 需注意 `D` 只能观测 **实际执行路径触发的未定义名称错误**，不是静态完备依赖证明；未执行代码路径中的依赖缺失仍可能漏检。固定训练oracle上的成功不代表隐藏测试正确。

## 反作弊与分析
- 外部测试隐藏用例仅在生成完成且SHA冻结后供Docker评分，不进入GRPO训练或提示。
- 独立种子43/44比较 SFT 与此前完整奖励 seed43/44；使用相同冻结外部题、相同采样顺序及预算。
- 对比 pass@1、四次至少一次通过、候选正确率、`NameError`、缺少 `typing.List` 注解导入回退、局部语义修复与回退。
- 不以单一训练种子的训练奖励上升或正向点估计宣称稳定泛化。
- 目前只设计并启动两种子训练；不能在真实生成和隐藏判题完成前填写外部结论。

## 可复现工程文件
- 奖励实现：`src/coderl_lab/train/exp024_reward.py`
- 训练器：`src/coderl_lab/train/exp024_7b_grpo.py`
- 配置：`configs/exp024_strict_seed43.yaml`，`configs/exp024_strict_seed44.yaml`
- 冻结脚本：`scripts/exp024_freeze_new96.py`
- 评测准备：`scripts/exp024_prepare_new96_eval.py`
- 顺序评测：`scripts/run_exp024_new96_eval.sh`
- 奖励测试：`tests/test_exp024_strict_reward.py`，当前4/4通过。
