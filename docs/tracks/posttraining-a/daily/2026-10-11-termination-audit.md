# 2026-10-11：TRAIN-009B SFT 原始输出终止率 / 生成预算实测

**状态（撰写时）：已固定真实输入、Qwen EOS 语义与测试方法，Tang 单卡已经启动生成，最终 32 个源提示的结果以 `results/train009b-termination-audit/summary.json` 在验收后另行归档，不预填正确数。**

## 研究为什么发生

TRAIN-009A 已验证“真实冻结RM → SFT初始Actor → 在线GRPO生成与奖励 → KL → 2步非零策略梯度 → 新 LoRA”；但v2原96-token 4/4回答截断被TRL mask导致没有梯度；v3把长度上限改256并在smoke显式允许截断参与更新，虽有2步非零梯度，8/8仍无自然EOS。不能据此推出完整回答质量收益，更不能把v3的宽松截断策略扩成正式训练。

## 输入及先验证据（不凭生成结果选题）

- 原SFT Policy：`Qwen/Qwen3-1.7B-Base@ea980cb0...` + 已正式训练的TRAIN-007A QLoRA Adapter SHA `d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d`，**不加载Reward Model，不进行训练或评测人类偏好**。
- 数据冻结：UltraChat SFT 4096train / 256 heldout，同Reward UltraFeedback train2048/heldout256的完整数据哈希此前已在GitHub固化。
- 16条TRAIN-009A初始Smoke train prompt（原顺序与seed42一致），加16条按seed42、SHA排序的**SFT heldout且与原train/RM任何splits完全不重叠**的测试prompt；一共32条，不能看到结果后替换失败个体。
- 真实Qwen3 tokenizer审计：`eos_token_id=151643 <|endoftext|>`，而`<|im_end|>=151645`属于不同的special token。二者**分别计数**，不把im_end事后自动当自然结束。官方Base generation config的默认EOS是151643。
- 训练prompt Token长度最大132；16条预选heldout中有**两个超长898/622 Token**。最初CPU Actions因强制`<=480`而失败。没有删题或换题，新增固定输入前处理：所有32条按相同规则**长度超480的保留最后480 Token**，以保护贴近`### Assistant:`的结尾上下文；同时逐条记录原长度、丢失前缀数量、是否截断。后来GitHub Actions `38144602512` 和CPU回归测试均通过。
- 先固定三个预算阈值`96, 256, 512`，每条只运行**一条max_new_tokens=512、temperature=0.8/top_p0.95/top_k0** 的真实采样轨迹，记录首次默认EOS/im_end位置；96、256窗口由同一轨迹前缀导出，这些不是独立的三次采样，更不能做成“多次随机试到成功”。
- 不把任何completion文本、隐私数据或hidden tests上传GitHub。只按split统计终止、截断、im_end出现次数、平均生成Token和长prompt截断数；只读GPU.
- 源代码 `src/coderl_lab/analysis/train009b_termination_audit.py`，脚本 `scripts/run_train009b_termination_audit.sh`，测试 `tests/test_train009b_termination_audit.py`。
- Tang `/home/you/projects/CodeRL-Lab-track-eos`基于GitHub固定commit `308f3dd8fcec70e831ed559d80f08ac7ce470ebb`的detached worktree，先完成8项专项CPU、GPU独占以及12GB磁盘最低余量检查。21:49+08开始真实GPU终止率生成任务。
- 第一版本地执行命令仅因尚未建`artifacts/`日志目录失败（未启动GPU）；已在外层安全创建后立即重新执行，日志、实际模型仍从GitHub冻结分支加载。无模型训练或checkpoint，磁盘使用率98%的其他项目数据不动。

## 后续决策边界

只在 **所有32条** 真实完成后报告split×token cap终止率。如果512有明显自然EOS，而96/256更多被截断，可以再独立冻结下一轮`mask_truncated=true`、只为EOS完整回答发放RM奖励的GRPO机械验收。不得针对这些EOS结果挑出看起来能自然结束的提示做成功示范。

验证自然终止比训练动态更弱：模型EOS并不等于内容正确、合理、人类偏好好，RM分数升高也不是独立价值评估。必须把EOS数据、奖励饱和/欺骗和独立holdout三种指标分开。


## 22:04：32条真正GPU生成全部完成，冻结结果

- 16条train与16条原本SFT heldout prompt均完成单次固定温度`temperature=0.8, top_p=0.95, top_k=0`采样，每条真实SFT模型至多512tokens。所有窗口从同一采样轨迹对应prefix导出，绝非挑后重复生成。
- **TRAIN前缀：**96tokens自然EOS2/16（12.5%）、256tokens 8/16（50.0%）、512tokens 13/16（81.25%）。
- **SFT heldout：**96tokens自然EOS1/16（6.25%）、256tokens 8/16（50.0%）、512tokens 12/16（75.0%）。
- **独立`<|im_end|>=151645`**：两组所有预算观测均0，不应随意解释成自然结束；默认`<|endoftext|>=151643`是本次唯一自然终止判据。
- 16个验证prompt中原始Token长度898、622的两个被固定`keep last 480`处理，没有删除、替换或挑选；train16都未发生截断。该输入处理对终止率有一定影响，必须在结果限制中写明。
- 真机GPU NVIDIA GeForce RTX4090 Laptop完成后进程退出，峰值预留显存`1616904192` bytes。逐行**只存哈希与Token统计**的trace SHA-256 `b24bd75fab09d87ceaf54a583d744bdb7e151563c71d50a0b83aadcd76b95bcc`，全量来源/模型版本身份从`frozen_inputs.json`冻结后未变；无原prompt与模型completion文本上传GitHub。
- 正式机器可读结果：`results/train009b-termination-audit/summary.json`；真实GitHub Source code/CPU工作流`38144602512`已通过，Tang本机8项专用测试通过。
- 初步判断：此前GRPO 96tokens及256tokens大量masked零梯度，并非只因学得RM没有奖励方差；生成预算是实际重要瓶颈。但上述统计是固定的32条（以及某些heldout左截断）上的终止率诊断，不证明512tokens下GRPO一定有非零梯度，也不保证自然EOS回答语义正确。
- 基于这一诊断独立冻结`TRAIN-009C`，保持`mask_truncated_completions=true`，只为真EOS响应支付学得RM奖励并尝试严格的2-step GPU机械验收，不以本次诊断选个别成功样本。
