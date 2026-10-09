# 2026-10-08 12:42–12:47：双主线并行执行追加记录

## A：GPU 与训练数据审计

- Tang RTX 4090 检查时空闲，原 ForecastLab NLI 进程已不占用计算显存。
- 独立工作树 `/home/you/projects/CodeRL-Lab-track-a` 使用 GitHub 代码；SFT 冻结训练文件从 `UltraChat@b7fe606...` 相同流式构建程序在 Tang 本地重建。
- 本地 full-source SFT train 4096 SHA256=`f868096a21eb37249d06d318fb56ab3b0e3e99c4db442e54c43d6f42b565888b`，heldout 256 SHA256=`a7f6e07f8569157fe5c3ae0deb65875c3427f9db3d6a5120379e1d9b10dae170`，均与 GitHub Actions 预先冻结快照完全一致。
- 安装隔离的 bitsandbytes overlay，不修改 B 分支或系统默认依赖。
- 对 **冻结全数据文件的前32条训练+8条保留验证**实际执行 2 optimizer step 的 NF4 QLoRA 冒烟，GPU 实测成功：train_loss 1.1635924；heldout eval_loss 1.5000250；peak reserved 3,932,160,000 bytes；adapter SHA256=`773e569a16a727d2961fdebd458821a38943beb2a8dbccbc757d31f0e31b85df`；GPU=RTX 4090 Laptop。
- 此 smoke 与此前使用前64源数据选样的 4-step smoke **不是相同的采样定义**，不能直接比较 loss。机器可读归档：`results/train007a-frozen-data-smoke/summary.json`。

## GPU 独占冲突（必须保留的失败记录）

- 本会话拟再启动完整 TRAIN-007A 时，`gpu_preflight` 返回 PID 3418262 已占用 GPU，严格拒绝了本次重复训练；
- 只读检查 PID 显示，该进程正是另一个执行链已启动的 **本项目同一 QLoRA 4096 训练**：`coderl_lab.train.sft --config configs/posttrain_a_qlora_qwen3_1.7b.yaml --data data/generated/posttrain-h4-v1/sft_train.jsonl`。
- 因此没有停止、修改、重启或并发复写正式 checkpoint；继续把这个已存在的进程视为正式执行者。
- 当前完整训练尚未产生经过本会话验证的最终 loss/适配器结果；不得提前宣称正式训练完成。
