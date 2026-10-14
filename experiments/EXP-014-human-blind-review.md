# EXP-014：三组真实生成回答的双人匿名盲审协议

本实验源自 CodeRL-Lab 的独立 HH-RLHF helpful-base 偏好测试。**本项目当前还没有任何人类盲审结果，不能把下面的量表当作已完成评级。**

## 审阅对象与遮蔽机制
- 人工审核只读取 Tang 工作目录 `artifacts/exp014/blind64/blind_review_packets.jsonl`，每条包含历史Human/Assistant轮次的相同上下文及随机排列的A/B/C三段真实模型回复；不能读取 `blind_assignment_key.jsonl`，不能看到模型SFT/BF16 DPO/FP32 DPO对应关系。
- 64条提示、192条回答；模型相同7B底座、相同提示、temperature0.7、top_p0.9、224新Token上限。所有候选由真实GPU推理生成，不以参考偏好标签挑出“最好”的样本。
- 每个任务由**至少两个相互独立的人类审阅者**分别做出 A/B/C/TIE 选择；不得在第一次选择前共享意见或查看另一个人的投票。
- 主要判断：①是否遵循并真正回答用户问题；②事实正确/合理与不臆造；③有用性与可操作性；④表达清晰、简洁和适当；⑤避免明显有害误导。任一候选可能都不理想，此时可选TIE并在独立笔记中说明，不强迫选所谓“最好模型”。
- 建议在A/B/C完全回答完毕、检查每人64条完整性和ID匹配之后，才由实验管理者读取密钥统计各模型胜出次数。不要提前做任何对模型身份的猜测。
- 当前 CLI 评分是**最佳回答选择**，不能冒充独立5维细项评分。今后如果确需维度评分，应提交新版本冻结方案，不能重写旧评价协议。

## Tang 本地运行命令
以下只需审阅者阅读，没有在 ChatGPT 中自动制造评分数据的接口。

```bash
cd /home/you/projects/CodeRL-Lab-track-exp014
PYTHONPATH=src /home/you/projects/CodeRL-Lab/.venv-gpu/bin/python \
  -m coderl_lab.analysis.exp014_blind_human_review review \
  --packets artifacts/exp014/blind64/blind_review_packets.jsonl \
  --rater-id independent_reviewer_1 \
  --output artifacts/exp014/blind64/reviews/reviewer1.jsonl
```

第二位独立审阅者只修改 `--rater-id independent_reviewer_2` 和 `--output .../reviewer2.jsonl`，在独立机器或者独立会话中完成即可。**审阅前必须确保输出目录已存在，rater文件原本不存在**；脚本拒绝覆盖评级文件。

两人完整完成后，再执行：

```bash
PYTHONPATH=src /home/you/projects/CodeRL-Lab/.venv-gpu/bin/python \
  -m coderl_lab.analysis.exp014_blind_human_review aggregate \
  --packets artifacts/exp014/blind64/blind_review_packets.jsonl \
  --key artifacts/exp014/blind64/blind_assignment_key.jsonl \
  --rater-a artifacts/exp014/blind64/reviews/reviewer1.jsonl \
  --rater-b artifacts/exp014/blind64/reviews/reviewer2.jsonl \
  --output artifacts/exp014/blind64/human_review_aggregate.json
```

## 统计输出与科研限制
- 逐任务审阅者一致率、未加权Cohen κ、TIE票数、两审阅者不一致案例数。
- 只有完成64题的两个人类审阅记录后才解除匿名映射，并报告每个模型的总体胜出票数与两人一致的“共识获胜”计数。不同审阅者投票不一致时不直接把其中一票当成真值。
- 这仍是**历史公开 helpfulness 任务、64提示、2个人类审阅者**的探索性比较，而不是新时间前瞻基准。LLM生产token的统计变化/偏好对数概率无法取代这些真实人工评审。
- 原始人类审阅文件、回复内容和匿名映射仅保留Tang本地；GitHub提交可复现的匿名化聚合、其SHA256与本协议，并按日期记录争议及重新审阅原因，不上传审阅者身份秘密。
