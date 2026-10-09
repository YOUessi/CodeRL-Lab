# CodeRL-Lab：从后训练概率重分配到可验证低 Margin 推理干预

> 2026-10-08 回顾性整理。本文把**已经存在于 GitHub 的决策、预注册和实验结果**连接成完整研究链，但**不是当时逐秒的思考笔记**。若某项前瞻性假设没有独立于结果的提交，不能把它冒充事前预注册。每段后面列明原实验材料、关键对照与尚未解决的问题。

## 研究的起点：SFT 为什么可能让更多采样反而变差？

最初 CodeRL-Lab 的目标是学习真实的大语言模型后训练（SFT、GRPO、DPO），并建立公开测试奖励与隐藏测试评测隔离。训练过程确实已经完成。让后续研究转向概率机制的发现来自 Qwen3-1.7B：

- 在同一 MBPP validation 90题的早期 n=4 评测中，Base→SFT 的 Pass@1：33.61%→36.67%；
- Base→SFT 的 Pass@4：64.44%→53.33%，下降 11.11 个百分点；
- 一个更强的基础模型，并未从这组小样本 SFT 上得到单调更好的有限预算采样覆盖。

**出现疑问：**是算法能力真正丢失，还是正确轨迹概率变得更低，所以小 k 不容易采出？

**决策：**先增大 k，不急于引入新训练目标。[EXP-006A](../../experiments/EXP-006A-qwen3-1.7b.md) / [PR #11](https://github.com/YOUessi/CodeRL-Lab/pull/11)。

## 01｜EXP-006B：支持集消失还是概率重新分配？

**对照：**固定 Base / SFT / SFT+GRPO，90题统一生成 n=16；对 k=16 时严格筛出的 Base 可解、SFT 不可解的9题扩展到64次，并验证前16个样本逐字节一致。

**结果：**SFT 在9题中的8题于 k=64 时重新找到正确答案；唯一剩余题的许多输出只缺 `import re`，添加该依赖后有17/64候选通过。这支持“行为概率分布/输出完整性重塑”而不是直接证明底层算法知识已经删除。

**边界：**有限 k 下无法在数学上证明支持集完全不变，模型遗忘仍可能存在。

**下一问题：**能否通过更细的执行奖励或偏好优化改善输出完整性、同时不牺牲覆盖？

证据：[EXP-006B](../../experiments/EXP-006B-large-k-boundary.md) / [PR #12](https://github.com/YOUessi/CodeRL-Lab/pull/12)。

## 02｜EXP-004A/B：训练信用分配与“正确偏好”替代解释

EXP-004A 把语法、依赖、运行时与公开测试分解为奖励。尽管有效更新增加，验证集并未产生可靠的正确率增益。

EXP-004B 则构造**验证过的最小标准库 import 修复**偏好对，并训练 DPO。虽然部分多样本覆盖有所改善，但**随机化偏好标签**也能取得类似增益，且语义 DPO 未可靠优于随机标签。

**新疑问：**DPO 的收益到底来自正确偏好语义，还是很小的训练更新改变了 SFT 已有的概率集中？

**决策：**先做负对照（no-op、反向标签、zero learning rate），不继续增加修复样本。

证据：[EXP-004A](../../experiments/EXP-004A-process-reward.md)，[EXP-004B](../../experiments/EXP-004B-runtime-repair-dpo.md) / [PR #15](https://github.com/YOUessi/CodeRL-Lab/pull/15)。

## 03｜EXP-004C：即使偏好没有信息，仍可能改变模型输出

**假设与控制：**

- 正向、随机、反向偏好；
- chosen == rejected 的 no-op，分别保留/关闭 LoRA dropout；
- zero learning rate 检查训练/保存本身是否引入差异；
- 任务和样本编号对齐，以相同随机种子比较输出。

**结果：**普通 no-op 在非零学习率下权重仍出现极小漂移，固定种子输出约30%发生变化；zero-LR 反而能和 SFT 1440/1440 逐字节复现。反向标签也可能恢复高 k 覆盖。

**对原假设的修正：**不能把任何 DPO 后的 Pass@k 改善自动解释成正确偏好被学到。此时仍然无法排除 DPO 特定优化器/更新几何因素。

**下一问题：**如果完全不跑 DPO，只对适配器施加同量级随机扰动，是否也会这样？

证据：[EXP-004C](../../experiments/EXP-004C-dpo-deconcentration.md) / [PR #17](https://github.com/YOUessi/CodeRL-Lab/pull/17)。

## 04｜EXP-004D/E：随机参数扰动和幅度扫描

**控制设计：**从相同 SFT LoRA 出发，按 no-op 的每个 tensor 范数制作随机方向扰动；使用多个固定 seed、0.25×/0.5×/1×/2× 幅度，统一输出与评价预算。

**结果：**多个方向的同量级随机扰动也能提高高 k 覆盖；较小的0.25×扰动就引起接近30%的固定随机种子输出变化。幅度-效果不是单调函数。

**解释范围：**证明 DPO 更新结构不是触发行为变化的必要条件，不证明任何噪声都能改进模型，也不能用3个 seed 推断所有随机方向。

**下一问题：**真正的大变化发生在模型概率分布本身，还是生成序列把微小差异放大了？

证据：[EXP-004D](../../experiments/EXP-004D-matched-norm-perturbation.md)、[EXP-004E](../../experiments/EXP-004E-perturbation-dose-response.md) / [PR #19](https://github.com/YOUessi/CodeRL-Lab/pull/19)、[PR #20](https://github.com/YOUessi/CodeRL-Lab/pull/20)。

## 05｜EXP-004F：区分初始 logit 变化与自回归级联

**设计：**增加确定性 greedy 对照、prompt末端分布 KL/TV、首次生成 token 分叉位置，以及多个 temperature 的固定种子采样。

**关键结果：**在12个 arm 中，prompt端 top1 仍100%一致，KL约0.001；greedy 最终轨迹仍约20%–34%发生分叉。首次分叉多数发生在 near-tie / low-margin 决策，温度越高只出现小幅额外行为变化。

**推翻的简单解释：**不是只有随机 sampling 才能放大微小扰动。

**新的可检验机制：**微小变化 → 局部 argmax 翻转 → 前缀反馈 → 自回归级联；stochastic sampling 额外放大。

**下一问题：**这些容易分叉的局部 token 能否事前预测？还是只是在看到分叉后找一个解释？

证据：[EXP-004F](../../experiments/EXP-004F-sampling-amplification.md) / [PR #21](https://github.com/YOUessi/CodeRL-Lab/pull/21)。

## 06｜EXP-004G：轨迹脆弱性预测，以及一项明确失败的假设

**预测实验：**利用 SFT reference 前128 token 的 margin≤0.05 密度预测12个随机扰动 arm 的 task-level 轨迹分叉。Spearman rho=0.3166，95% bootstrap CI [0.1115,0.4991]；控制长度后仍为正。

**被反驳的解释：**“SFT 全局增加低 margin 密度”并不成立。对照 Base vs SFT 后发现低 margin 比例反而从3.58%降至2.29%。

**科研决策：**必须放弃“整体 confidence 变差导致不稳定”这条简单叙述；聚焦**稀疏的局部瓶颈**及其是否具有因果作用。

证据：[EXP-004G](../../experiments/EXP-004G-trajectory-susceptibility.md) / [PR #22](https://github.com/YOUessi/CodeRL-Lab/pull/22)。PR #22 的单独提交显示 Phase A 结果之后才注册 Phase B，随后记录负结果与机制修正。

## 07｜EXP-004H：从相关到局部因果干预

**操作：**固定参考序列中第一个 margin≤0.05 的 token，对原始 top1 加 +0.25 (stabilize) 或 -0.25 (destabilize)，并在高 margin 位置设置强度匹配的控制组。

**结果：**stabilize 提高完整 greedy 轨迹恢复概率；destabilize 大幅增加分叉；同强度 high-margin control 基本不产生变化。

**必须区分：**改变轨迹 ID 不等于回答正确。因此下一步评价应从字符串 exact match 转向真正可执行的任务正确性。

证据：[EXP-004H](../../experiments/EXP-004H-local-bottleneck-causality.md) / [PR #23](https://github.com/YOUessi/CodeRL-Lab/pull/23)。

## 08｜EXP-004I/J：从局部因果到真实正确率

EXP-004I 在已有轨迹上评估 hidden correctness，发现 reference 错误时 destabilize 有机会解锁正确替代轨迹，但 reference 正确时也可能破坏原答案。

**难点：**推理时不知道 hidden correctness。不能用 hidden 结果决定是否干预，那是信息泄漏。

**可观察替代：**只使用 public tests 的通过/失败作为 Gate；在失败且存在 low bottleneck 时干预。这一策略在离线分析中方向为正。

EXP-004J 移除扰动 adapter，在 SFT 自己上执行两阶段推理。90题出现 +3.33pp，但95% CI=[0,+7.78]，不满足预注册的区间下界严格大于0标准。

**决策：**停止在多次用于开发的90题上调阈值，冻结规则并转独立测试。

证据：[EXP-004I](../../experiments/EXP-004I-bottleneck-correctness-causality.md)、[EXP-004J](../../experiments/EXP-004J-verifier-gated-bottleneck-escape.md) / [PR #24](https://github.com/YOUessi/CodeRL-Lab/pull/24)、[PR #25](https://github.com/YOUessi/CodeRL-Lab/pull/25)。

## 09｜EXP-004K：真正冻结后的500题 held-out 复核

冻结 SFT、greedy、window=128、low≤0.05、high≥0.20、logit bias=-0.25、public-fail Gate、20,000 task bootstrap / seed42。

- 原始 SFT：207/500=41.40%；
- gated-low：215/500=43.00%；
- +1.60pp，95% CI=[+0.40,+3.00]；wrong→correct=10，correct→wrong=2；
- high-margin control=0pp；
- ungated always-low=0pp（10 rescued，10 harmed）。

严格意义：本研究获得在同一 **MBPP 家族**、未参与方法设计的500题上的独立复制；仍不能说已经证明跨任务类型、跨数据分布泛化。

证据：[EXP-004K](../../experiments/EXP-004K-heldout-verifier-gated-replication.md)、[results/exp004k/summary.json](../../results/exp004k/summary.json) / [PR #26](https://github.com/YOUessi/CodeRL-Lab/pull/26)。

## 10｜EXP-004L：LiveCodeBench v6 外部分布（当前阶段）

冻结上述所有策略规则，从 MBPP 切到 2025-01～04 的 LiveCodeBench v6 175题，包含 AtCoder stdin 与 LeetCode functional 题。

- 数据集 + evaluator 的完整 commit 和 SHA 固定；
- 12-task 跨两类格式的工程 smoke 已通过，baseline 与 gated-low均1/12，属于流程测试而非性能结论；
- 175题官方私有评测已完成：baseline 14/175 (8.00%)、gated-low 15/175 (8.57%)、gated-high 14/175、always-low 15/175；primary +0.57pp，95% CI [0,+1.71]，**未达到预注册显著性标准**；
- private tests 只在 runner/public gate/second-pass 冻结后执行。竞赛任务 hard 0/80，强烈提示小模型能力地板，但不能确定它解释了全部迁移失败。该负结果要求后续独立模型/任务预注册，而不能直接调此测试集参数。

证据：[EXP-004L](../../experiments/EXP-004L-livecodebench-verifier-gated.md)、[正式175题结果](../../results/exp004l/summary.json)、[12题 smoke](../../results/exp004l-smoke/summary.json) / [PR #27](https://github.com/YOUessi/CodeRL-Lab/pull/27)。

## 研究方法与复现索引

- 假设/预注册：各 `experiments/EXP-*.md`，部分预注册可通过 PR 的独立提交时间证明。
- 结果和图表入口：`results/exp004{b,c,d,e,f,g,h,i,j,k}/summary.json`；B v6 进行时另见 `results/exp004l-smoke/summary.json`。
- 真正执行代码：`src/coderl_lab/analysis`、`scripts/run_exp004*.sh`、`tests/`。
- 逐日工程/研究日志：`docs/daily/2026-10-07.md`、`docs/daily/2026-10-08.md`。
- 大体量原始 rollout/Token trace/完整模型 checkpoint 不直接上传 Git，仅在实验环境保存；需要定期校验是否仍存在并建立带SHA的可访问目录索引。
- **不要倒果为因**：最初探索中具有研究自由度，单一 validation 多次复用，最终 K 的独立复制和 L 的跨分布测试正是为了控制这些风险。

## 与 Track A 的边界

本发现链是 **Track B：机制科研**。Track A 已在独立分支 `feat/posttraining-track-a-data-qlora` 启动真实多来源 SFT/DPO 训练数据和 QLoRA/LoRA 对照，不将 B 的私有测试加入训练。两条线共享模型工程基础设施，但实验目标、数据使用和 GPU 时段独立。
