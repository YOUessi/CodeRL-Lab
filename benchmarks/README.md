# Benchmark 数据格式

当前采用函数级 Python 任务，每行一个 JSON 对象。

必须字段：

- `task_id`：唯一任务 ID；
- `prompt`：自然语言任务；
- `entry_point`：目标函数名；
- `public_tests`：训练可见测试；
- `hidden_tests`：仅评测测试。

可选字段：

- `starter_code`；
- `metadata`。

测试用例格式：

```json
{
  "name": "case_name",
  "args": [1, 2],
  "kwargs": {},
  "expected": 3
}
```

当前 JSON 测试协议只覆盖可 JSON 序列化的输入输出。复杂对象、浮点容差、异常期望、文件输入输出将在正式数据集阶段扩展。
