# 2026-10-10：TRAIN-009A 在线 PPO-Clip / RLHF 实操闭环

## 为什么新增本实验

之前完成的TRAIN-007A/B SFT、TRAIN-007C DPO、TRAIN-008A独立Reward Model训练，都属于离线监督或偏好优化。TRAIN-007D的252对同口径SFT-vs-DPO评测说明DPO虽然改变全部252个偏好对的模型logprob margin，却没有改变任何chosen/rejected排序。TRAIN-008A Reward Model对固定UltraFeedback256 heldout观察到+11.72pp，但并非PPO。

本次目的是第一次让**已训练SFT Policy → 在线生成 → 独立冻结Reward Model → 初始SFT Reference KL → 带Critic的GAE → PPO-Clip策略/价值头反向更新**真正连接在一起，并通过 Tang RTX 4090真机闭环。优先完成真实、可审计的短小GPU烟雾，而非编造“完整RLHF提升”。

## 10:58—11:12：环境、库和数据防泄漏

- Tang在线，显卡RTX 4090 Laptop 16GB，检查时无其它CUDA模型作业，剩余系统盘约47GB/95%已用，后续不复制巨大权重至GitHub；仅真机测试调用Desktop Commander。
- 软件固定：torch2.10.0、transformers5.18.0、TRL1.14.1、PEFT0.21.2、独立bitsandbytes0.50.2。实际环境中 `from trl import PPOTrainer` 返回`ImportError`；**不能假装直接使用TRLPPOTrainer**。选择自己用PyTorch实现明确的PPO-Clip逐Token比率、可训练状态Critic和GAE，数学参考函数、测试和GPU实现都有仓库源码。
- 从 `feat/posttraining-reward-model` 独立创建 `feat/train009a-online-ppo-clip-rlhf`，Draft PR#32；代码/配置/日志全部以GitHub为唯一事实源。
- 冻结SFT policy Adapter SHA `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d` 和已训练RM Adapter SHA `586fb4cb21bb3408507d6e179d1bd858aca35d7c3e46b1dbef84234d3adbd28c`；真实Tang逐文件重新计算均一致。
- 直接审计完整UltraChat SFT train4096、SFT heldout256、UltraFeedback RM train2048、RM heldout256四份文件，发现UltraChat train与RM train仅**1条prompt重叠**、与RM heldout **0条**；生成器仍严格排除RM train和两份heldout全部prompt，之后按 `SHA256(seed42|PPO|promptSHA)`选**8条训练 +4条预留probe**。无标签泄漏，未访问任何LCB private/MBPP hidden。
- 新增 `ppo_math.py`、`ppo_data.py`、`ppo_online.py`、`configs/train009a_ppo_clip_rlhf_smoke.yaml`、GPU受保护脚本及CPU单元测试。真实H4数据/数学验证[GitHub Actions](https://github.com/YOUessi/CodeRL-Lab/actions/runs/38019590804)成功。
- 在Tang创建仅真机实验用的detached worktree `/home/you/projects/CodeRL-Lab-track-ppo`，初始GPU执行GitHub SHA `9e48f2ece93ce22cb6038bae3d54bf8538573632`；真实数据、权重SHA和10项本地CPU专项测试通过。

## 11:12—11:14：真实训练成功，检查LoRA确实更新

- 运行 `scripts/run_train009a_ppo_clip_smoke.sh`，由同一单GPU互斥锁保证不与其他训练抢卡；初始policy/reference均加载相同SFT Adapter，实际old/ref action-logprob max absolute difference为 **0**。
- 真正完成4个on-policy rollout batches：每批2独立prompt、每prompt采2份回答，**16个生成回答、512个新Token**。
- 使用RM对这16个采样生成实时打分、逐token计算KL=-β(logπ_old−logπ_ref)并在终点加scalar Reward，带状态价值头/GAEλ0.95、γ1；PPO-Clipε0.2，每批2轮优化共**32个真实optimizer mini-batch updates**；不是无Critic的GRPO或单次DPO。
- Tensor侧价值头`Linear(2048→1)`真实经过梯度更新，保存后 weight L2 `0.04177734`、bias `0.00032598`，都非零。
- 真机读取old SFT和新PPO两个safetensors：共同的**392个LoRA权重张量全部发生数值变化**，最大绝对元素差0.0001477972；原初始SFT SHA未改变，冻结RM SHA未改变。
- 策略LoRA Adapter SHA `0b302778bcd44a2c01830c93e749eb098b9be7f75ff84cb81d314856b50bec70`；Critic head SHA `a946efd327a9203e0dae8b31971f5502ea519807ffc8cd4d1735ab41fc5c73dd`；optimizer/RNG state SHA `241496b5efd00fc60095266d08afae23189837c0c56d48992718a43cbaacba86`，全部从真实文件重新计算且与总结一致。optimizer含394个非空参数状态，CPU/CUDA RNG存在。
- 真机wall 76.12秒，peak GPU reserved=5,165,285,376 bytes；平均PPO clip fraction=0.0048828125，sampled KL均值=-0.00072246。**单批有限采样KL差可能为负**，它是单条轨迹的logprob差样本均值，不是精确全分布KL散度；并不能因为接近零就宣布长期policy无漂移。
- 训练每批RM标量均值 [1.1506,1.6484,-0.6968,1.9238] 跨不同prompt，不是固定验证样本上的可比曲线，绝不可说“Reward单调提升”。
- 完成全部4步并通过 `policy_adapter/...`、`value_head.safetensors`、`optimizer_state.pt`、`run_summary.json`完整性门禁。机器可读主结果：`results/train009a-ppo-clip-smoke/summary.json`；原始模型状态留Tang，GitHub仅保留聚合/哈希/选择prompt SHA。
- **不允许声称**这是正式大规模PPO、证明RLHF提高人类偏好质量、或者reward头是无偏人类评判器。

## 11:19：固定4个未训练probe的真实策略对照（负结果）

- 在GPU得到数据之前，先在 `experiments/TRAIN-009A-online-ppo-clip-reward-model.md` 固定probe评测设计：4条预先通过SHA选择、从未用于PPO gradient的UltraChat train-source提示，同样128prompt+32completion greedy decoding，对比原SFT参考与训练后的PPO策略，再用**同一冻结RM**打分。该probe没有标注，不是外部独立human test。
- `analysis/train009a_probe.py`、`scripts/run_train009a_reward_probe.sh`和CPU检测固定4条、拒绝标签访问与非有限值；Tang本地12项相关单测和GitHub CI通过。
- 真机约11:19完成：4条中**2条greedy回答文本SHA改变**；SFT参考的平均RM打分 **0.8720703125**，PPO的平均分 **0.6318359375**，差 **-0.240234375**；两臂平均回答长度均32token。GPU结束后无存活CodeRL计算任务。
- 这是一份实测**负结果**：训练能确实移动LoRA权重、改变部分模型输出，但短期内没有证据证明奖励改善，这批固定probe的RM代理分数反而下降。只有4例，同一个RM同时用于策略训练与本次评分，**既不能证明人类质量变差，也不能证明泛化无效/有效**。
- 结果按冻结名称永久提交 `results/train009a-ppo-clip-smoke/fixed_4_probe_summary.json`，不上传真实对话文本。**不在相同4条probe上后验调整KL、PPO clip、prompt或reward scale来挑一个更好成绩**。

## 遇到的问题、如何解决以及下一步

- TRL固定版本没有可用PPOTrainer → 明确编写可复现的标准PPO-Clip+value/GAE PyTorch更新，而非贴错框架名。
- RM训练prompt与SFT训练prompt存在一条交叉 → 源数据构建明确剔除全部RM train和两套heldout提示，再固定SHA确定抽样，不挑结果。
- 之前DPO只报告参考相对奖励，不证明优于初始SFT → PPO本身重新显式加载SFT reference，并严格验证新旧初始一致，KL为逐token采样估计。
- 训练过程RM数值随新prompt大幅波动 → 不把不同prompt的batch均值认作因果奖励提升，设计未训练probe，并据实保存负结果。
- 正式下一步应是**独立外部偏好/任务指标、强策略基线、reward hacking防护及更多重复试验**。当前4步烟雾明确不应进入正式论文主效果表。若继续训练更多steps或改动PPO策略，必须另建冻结试验/训练分支并在新盲测上检验。
