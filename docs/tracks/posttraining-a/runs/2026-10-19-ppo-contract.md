# PPO 采样与有限时域契约修复（CPU 审查）

## 基线与历史边界

审查固定基线为 `774270356b20fefb455260cc088f8865b643bf9d`：

- [ppo_online.py](https://github.com/YOUessi/CodeRL-Lab/blob/774270356b20fefb455260cc088f8865b643bf9d/src/coderl_lab/train/ppo_online.py)
- [ppo_math.py](https://github.com/YOUessi/CodeRL-Lab/blob/774270356b20fefb455260cc088f8865b643bf9d/src/coderl_lab/train/ppo_math.py)
- [历史配置](https://github.com/YOUessi/CodeRL-Lab/blob/774270356b20fefb455260cc088f8865b643bf9d/configs/train009a_ppo_clip_rlhf_smoke.yaml)

历史运行是 Qwen3-1.7B CUDA 工程 smoke：4 批、16 条轨迹、32 次 mini 更新。
本变更不修改该配置、任何历史结果或历史结论，也不把历史运行重新认定为严格满足本契约。
本轮未运行 GPU，不新增质量提升、泛化或正式 RLHF 效果结论。

本报告与实现由独立分支 `fix/ppo-contract-20261019` 交付；最终实现提交号、远端完整 CPU 测试及独立审查由协调者补充，不能把下列本地部分测试当作全套验证通过。

## 已确认的问题与保留的正确行为

1. 历史 generate 使用 temperature=0.8、top_p=0.95，而 old/new/reference logprob 使用原始 logits 的 log_softmax。采样行为分布与 PPO 概率比、采样 KL 的分布不一致。且没有显式关闭 top-k 或模型携带的其他 generation defaults。
2. 单轨迹当前 generate 通常在 EOS 停止，并不天然产生 padding；旧代码缺少明确的 EOS 后缀裁剪契约。新代码防止未来附带的 EOS 后 padding 进入奖励、优势或损失，不宣称历史已观测到 padding 污染。
3. 原 scorer 的 `prompt_length-1:-1` 位移正确，value 是采取对应 response action 之前的状态值。原 GAE 对其已有的零终值有限时域目标计算正确。本次不引入末状态 critic bootstrap。
4. 原 GAE 文档把一般 lambda 情况称为 Monte-Carlo 不够准确；只有 lambda=1 才退化为相应折扣回报。

## 选择的最小一致契约

采用原始全词表 softmax 作为行为策略及 PPO 优化策略。新配置 temperature=1、top_p=1；代码用全新 `GenerationConfig`，显式 top_k=0，且 `generate(..., use_model_defaults=False)` 禁止继承模型上的采样温度、强制 token、重复惩罚等默认值。prompt attention mask 全为 1，因为每次只处理一条无 padding prompt。

old/new/reference 都继续由原始 logits 的 log_softmax 计算。这样无需在每次更新中重建会变化的 nucleus 支持集，也避免 old-policy 样本落在 new-policy 的零概率区域。此次有意不支持 temperature/top-p 变换后的策略优化；旧的 0.8/0.95 配置被 `load_config` 明确拒绝，避免继续以不一致契约训练。不是静默覆盖用户填写的温度。

**迁移：**历史 YAML 保持原样供重现和审计。新 CLI 默认 `configs/train009a_ppo_raw_policy_contract.yaml`；调用旧 shell wrapper 或显式传入旧 YAML 会失败，需要显式切换新配置。新输出目录也不同，不能覆盖旧 smoke。若确需重现历史缺陷，使用上述固定历史提交。

目标定义为生成长度至多 H 的有限时域响应任务。EOS 或 H 都是任务终点，终点状态 value 为 0，最终 reward-model 分数加在最后一个有效动作上。达到长度上限不补写 EOS。这个选择保留原有目标并将它写明；它不等价于一个希望继续生成的无限/长时域任务被外部 time limit 截断。

每条 response 在首个 EOS **之后**截断，EOS 本身保留（即使 pad token 与 EOS 相同）。没有 EOS 时必须恰好达到 H，否则拒绝未知的提前终止。prompt 不进入 action slice，EOS 后 padding 不进入任何训练数组；有效 token mask 由无 padding 的 response slice 表达，不另外维护可能错位的布尔 mask。奖励、GAE、actor loss、critic loss 使用同一 slice。输出记录 eos/length 终点计数与契约版本。

## 确定性算例与测试状态

测试文件：`tests/test_ppo_contract.py`。无远程权重、无 CUDA 要求；tiny GPT2 由随机初始化的本地小配置构造。

- EOS 后缀 `[4,2,2,2]` 保留 `[4,2]`；首 token 是 EOS 时仍保留一个动作。
- 长度上限 `[4,5]` 保持原样，原因记为 length；最后一个位置恰好为 EOS 时记为 eos。
- 非零 value 手算：reward=[0.2,2]，value=[0.5,0.6]，gamma=.9，lambda=.5，delta=[.24,1.4]，advantage=[.87,1.4]，return=[1.37,2]。单动作 reward=2、value=9 时 advantage=-7、return=2，末状态没有 bootstrap。
- KL shaped reward 只在最后有效动作加 scalar reward，并包含 EOS 动作的 KL 项。
- Torch scorer 用已知 logits 与 hidden states 验证前移一位、prompt/末状态梯度为零、EOS 动作梯度非零，以及初始 ratio=1。
- 真实 tiny-transformer generate 故意污染模型 generation defaults，比较实际采样 scores 与生产 scorer 的 old logprob，并验证整个词表 scores 有限、初始 ratio=1。
- 生产 rollout 路径的确定性 EOS fixture 验证评分前先去掉 post-EOS padding，空文本 reward 落在 EOS 动作上。

本地命令：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_ppo_contract.py -v
```

红阶段：旧代码 8 项，3 failed、3 passed、2 skipped。失败均对应缺失的契约校验/EOS/长度处理。
绿阶段最终：`Ran 10 tests ... OK (skipped=4)`，即 6 个纯 Python 测试通过，4 个依赖 torch/PyYAML/transformers 的测试未运行。该环境没有这些依赖，不能声称集成路径已验证。

```bash
python -m compileall -q src/coderl_lab/train/ppo_online.py src/coderl_lab/train/ppo_math.py tests/test_ppo_contract.py
# exit 0
python -m pytest -q
# exit 1: No module named pytest
```

协调者应在具备项目依赖的完整固定快照运行：

```bash
CUDA_VISIBLE_DEVICES='' .venv-gpu/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-gpu/bin/python -m unittest discover -s tests -p test_ppo_contract.py -v
```

## 可运行的后续训练入口（本轮不执行）

```bash
python -m coderl_lab.train.ppo_online \
  --config configs/train009a_ppo_raw_policy_contract.yaml \
  --data-root data/generated/posttrain-h4-v1 \
  --sft-adapter artifacts/posttrain-a/train007a-qlora-ultrachat \
  --reward-adapter artifacts/posttrain-a/train008a-reward-model \
  --output-dir artifacts/posttrain-a/train009a-ppo-raw-policy-contract
```

仍要求固定 1.7B 模型、固定 SFT/RM adapter SHA 和数据排除规则；不是 7B 配置。新目录必须不存在。已有 provenance 会包含新配置 SHA 与 `raw-softmax-finite-horizon-v1`。

## 残留边界

- 尚需完整 CPU 测试和独立审查；量化 CUDA cached generation 与 noncached scoring 的微小数值误差没有在本轮测量。CPU 测试不能代替新 CUDA smoke。
- 代码当前是一条无 padding 轨迹一次更新；若改成多条 padded batch、多个 EOS id、stop strings 或外部截断，必须扩展契约和测试，不能直接沿用当前 slice 假设。
- gamma=1 是当前默认；有限时域 value head 不显式接收剩余步数，这可能限制 baseline 拟合，但不改变终点零值定义。
- 采样 KL 是单样本 log-ratio，单条样本可为负，并非精确非负 KL。RM 仍是代理分数，不是正确率或人类偏好验证。
- 现有 Torch PPO 对 log-ratio 使用 [-20,20] 数值夹限，而纯 Python surrogate 遇到超过范围会拒绝。此次未改动该既有极端 off-policy 保护差异；不能据此声明两个实现在所有极端输入等价。
- 没有改变 episode 内 token mean / episode 顺序更新的权重约定，也没有增加 resume 或正式训练预算。
