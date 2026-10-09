# 2026-10-10：TRAIN-008A 真实 Reward Model 2048/256 全量训练

## 前一阶段检查点

- TRAIN-007C 真机完整NF4 DPO已经在2026-10-09 18:10完成251/251 optimizer steps，有252条有效heldout偏好验证；源结果 `results/train007c-dpo-formal/summary.json`。
- TRAIN-008A 之前仅做过32train/16heldout、4优化步的GPU smoke，训练前后准确率都62.5%；不能以它代替正式数据规模实验。
- 全量 Reward Model 的数据路径为共享冻结 `HuggingFaceH4/ultrafeedback_binarized@daee4cffe8bea93b1cd7874b01c5a6a4ae9dcaf2`，2048 train、256 heldout，train SHA `945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77`、heldout SHA `9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450`。
- GPU实训前依据真实 Qwen3 tokenizer对所有对打包审计完成（见 `results/train008a-token-audit/summary.json`），结果中没有chosen/rejected完全相同编码输入。

## 03:11—03:12：正式训练实际启动与唯一GPU占用

- Tang `/home/you/projects/CodeRL-Lab-track-rm` 在03:11确认GPU仅桌面进程，15.398GB显存空闲。磁盘剩余约53GB、使用率95%，未出现空间不足故障，但检查点会轮换保存4份，禁止无故复制大型权重。
- `git fetch origin feat/posttraining-reward-model`，核实本机detached工作树精确GitHub提交 `89cd3a413a0cca1ecc9046bbc776fd2fd8cb9a7b`。工作树无修改，16项CPU专项测试、真实2048/256偏好数据manifest/SHA、single GPU `gpu_preflight --min-free-mib 12000`和 `flock`门禁全部通过。
- 真正启动GPU正式命令：`MODE=formal PYTHON=/home/you/projects/CodeRL-Lab/.venv/bin/python QLORA_EXTRA_PYTHONPATH=/home/you/.cache/coderl-track-a/deps bash scripts/run_posttrain_a_reward_model.sh`。
- 训练源：`src/coderl_lab/train/reward_model.py`，Qwen3-1.7B-Base sequence-classification head（`score.weight`是新初始化、属于预期，不是预训练权重缺失造成的训练失败），NF4/LoRA r16、score可训练；1epoch、2048真实偏好对、grad accumulation8、256 optimizer steps。记录 `artifacts/posttrain-a/train008a-reward-formal-20261010.log`，模型根目录 `artifacts/posttrain-a/train008a-reward-model`。
- 该实验优先进行**256 heldout训练前**排序准确率/成对loss/mean margin验证，再正式更新梯度；每32 optimizer step保存包括模型/optimizer/scheduler/RNG的checkpoint，禁止把未完成权重视为最终结果。

## 03:13—03:21：实际训练中期检查

- 03:13真实完成预训练头权重的模型加载，CUDA分配约3.4GB；训练前256 heldout基线 `initial_heldout_metrics.json` 保存成功：
  - 准确率 **0.44921875**（115/256），zero ties；
  - 平均margin `-0.1006470374`；
  - 平均pairwise loss `0.8597028020`。
- 03:21已实际执行 **56/256** optimizer steps，`checkpoint-32`存在；GPU显存占用约3.79GB，尚无其他模型任务。
- **目前正式训练仍在运行**，不能报告最终256验证准确率或模型泛化收益。无需对结果进行调参或重启。

## 并行但不占用GPU的工作

- 基于母分支A的独立 `exp/train007d-matched-sft-dpo-preference-eval` (Draft PR#31) 开发同一252有效UltraFeedback heldout偏好下SFT与DPO的概率配对比较。用单张4090的前向推断，**不与本次Reward训练并发**。
- 该分析源数据、两份adapter权重SHA和Qwen3 tokenizer 256→252过滤预检已在GitHub CPU CI/ Tang本机通过。Tang已启动受条件保护的本地队列，只有完整RM256步、真实256 heldout以及Adapter SHA完成后才会尝试GPU分数。
- 结果若失败或没有提升，仍按原始冻结标准归档，不回过头按heldout结果挑参数。


## 03:23：可恢复状态不是纸面承诺，已经验收实际Optimizer文件

- 训练达到 **72/256** optimizer steps，已保存 `checkpoint-32` 和 `checkpoint-64`。
- 在真实Tang上调用项目源代码 `validate_resume(output_dir, checkpoint-32, identity)`，重新读取全部冻结config/train/heldout源SHA及模型revision，身份校验通过。
- 再对真实 `checkpoint-32/state.pt` 只读检查：`step=32`、`processed_pairs=256`、优化器非空state 393 entries、CPU RNG/CUDA RNG及scheduler state均存在；同目录Adapter `adapter_model.safetensors` 约69.8MB。未修改运行中的模型、权重或优化器。
- GPU仍由唯一Reward Model训练进程使用，未启动 TRAIN-007D/其它并行CUDA任务。
- 完整模型与验证结果尚未产生，本条记录只证明中途**可安全检查并具备恢复状态**，不宣称发生了实际续训、也不以中途指标决定调参。

## 03:49 训练结项，05:03 真机复核与GitHub结果归档

- 实际 `TRAIN-008A` GPU 训练于2026-10-10 03:49:42（UTC+8）完成 **256/256 optimizer steps**。最后4个checkpoint为160/192/224/256，且每32步保存包含optimizer/scheduler/CPU+CUDA RNG的中间状态。
- 输入训练偏好2048、完整独立heldout256，来源revision和训练/验证 SHA 不变，未为了提高准确率挑样本。单epoch train mean pairwise loss=`0.7108690377608582`，wall=2222.10秒，peak reserved=3,038,773,248 bytes。
- 在**相同256对偏好**上，训练前随机初始化奖励头选对115对（44.921875%），训练后选对145对（56.640625%），观测提升**30/256=+11.71875个百分点**；pairwise heldout loss `0.8597028020→0.6970158364`，mean reward margin `-0.1006470374→+0.1465626673`。
- 最终Adapter SHA `586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c` 直接从Tang实际 `adapter_model.safetensors` 再计算并和 `run_summary.json` 逐值相同，属真实模型产物。
- 05:03再次检查Tang GPU空闲且RM Python进程已退出；没有失败重启或冒充未完成。
- 正式机器可读实验记录入GitHub `results/train008a-reward-model-formal/summary.json`。模型、优化器/原始偏好内容仍在Tang本地；GitHub是代码、实验配置、过程与安全聚合结果的唯一事实源。
- **科学边界**：单种子、一份heldout与新初始化分类头；不能直接称“提高真实人类偏好质量”，也不能当成完整PPO/RLHF策略训练。下一步应独立设置新的盲测数据/不同seed后方可讨论泛化。
