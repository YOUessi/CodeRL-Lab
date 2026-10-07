# EXP-004K：Held-Out Verifier-Gated Bottleneck Replication

## 状态

- 分支：exp/heldout-verifier-gated-bottleneck
- 数据：MBPP test 500 tasks
- 规则：完全冻结自 EXP-004J
- held-out runner：实现中
- smoke：待运行
- 正式500题：待运行

## 研究目的

EXP-004J 在反复用于机制开发的 MBPP validation 90题上观察到：

- SFT baseline hidden accuracy：40.00%
- public-gated low-destabilize：43.33%
- +3.33 pp
- wrong→correct=3
- correct→wrong=0
- high-margin control=0 change

但 primary 95% bootstrap CI=[0,+7.78]，没有严格满足预注册的 CI lower bound >0。

为了避免继续在同一 validation 上调参，本实验冻结全部规则，直接在此前没有参与 EXP-004F-J 方法设计的 MBPP test 500题上做 held-out replication。

## 冻结规则

不再修改：

- model：Qwen3-1.7B-Base + EXP-006A SFT adapter
- greedy decoding
- max_new_tokens=512
- primary window=128
- low threshold=0.05
- high threshold=0.20
- bias=-0.25
- gate：SFT baseline public tests 未全部通过
- gated-low / gated-high / always-low 定义
- task bootstrap iterations=20,000
- bootstrap seed=42

## 数据

固定：

- data/generated/mbpp-v1/test_tasks.jsonl
- 500 tasks
- task_id unique 500/500
- public tests 500/500 non-empty
- hidden tests 500/500 non-empty

test split 在本实验开始前没有用于 threshold / bias / gate 设计。

## 无 hidden 泄漏的数据流

### Phase 1：SFT baseline + margin trace

GPU 只生成：

- baseline raw completion
- reference token ids
- first128 margin profile
- low/high intervention positions

不运行 hidden tests。

### Phase 2：public-only gate

只对 baseline code 运行 public tests。

生成 gate：

public_fail AND low_position exists

该阶段的 artifact 不包含 hidden outcome。

### Phase 3：SFT second-pass intervention

保持同一 SFT adapter：

- gated_low_destabilize：gate=true 时 low bottleneck -0.25
- gated_high_destabilize：gate=true 时 nearest high-margin -0.25
- always_low_destabilize：所有 eligible task low bottleneck -0.25
- baseline：不改

### Phase 4：最终 outcome

所有生成完成以后，才统一运行 hidden tests。

hidden outcome 不参与 gate、position 或 bias 选择。

## Primary hypothesis

全500题：

gated_low hidden correctness - baseline hidden correctness

预注册成功标准：

- paired task bootstrap 95% CI lower bound >0

## Secondary controls

### Position specificity

gated_low - gated_high

### Gate necessity

gated_low - always_low

同时报告：

- wrong→correct
- correct→wrong
- public-pass task 是否被 gated policy 改写
- eligible / gate-triggered task 数

## 解释

如果 held-out primary CI >0：

说明 public-test-gated low-margin escape 在未参与方法开发的数据上得到复制。

如果点估计正但 CI 跨0：

只能认为方向一致但证据不足。

如果无效或负向：

EXP-004J validation 结果应视为开发集特定现象，不继续优化该 decoding rule。
