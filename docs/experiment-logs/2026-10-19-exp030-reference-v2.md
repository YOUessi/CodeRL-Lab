# 2026-10-19｜EXP030参考环境兼容性修订

## v1真实结果与原因

GitHub提交d51bd7aeeb18443cb24188f53ae664a0677148eb对应460个文件已在Tang逐Git blob核对。数据压缩源、公有与私有文件哈希一致；7项CPU契约测试通过。参考门禁实际执行87个独立隔离容器：86通过，DS1000/455因TypeError失败。失败结果保存为`results/exp030_reference_preflight_v1.json`，原目录不改写。

该题的原始reference_code与checker生成期望值都调用`np.select`，choices含字符串，default为NaN。固定镜像NumPy2.2.6将Python标量视为弱类型，无法合并该字符串与浮点默认值。一个仅含NumPy数组的独立容器算例复现了同样的TypeError；把choices与default先转为数组得到预期`['nan','XX']`。这是参考环境不兼容，尚未生成或评分任何模型答案。

官方[NumPy1.26.4 select源码](https://github.com/numpy/numpy/blob/v1.26.4/numpy/lib/function_base.py)在dtype推导之前对choices与default调用asarray。[NumPy2迁移指南](https://numpy.org/doc/2.1/numpy_2_0_migration_guide.html)记录了标量类型提升改变；[2.2发布说明](https://numpy.org/doc/2.3/release/2.2.0-notes.html)说明旧promotion-state开关已移除。镜像中的实际探针也确认该开关不可用。

## v2受限兼容适配

保留同一镜像、全部87题、原始prompt/reference_code/code_context和全部资源限制。在每个参考与模型容器中统一安装一个`np.select`兼容包装：将choicelist的元素与default先转为NumPy数组，再调用原select。这恢复该函数的旧式输入数组化规则，**不声称完整降级或复现NumPy1.26环境**。没有按题号分支，也没有修改任一原始checker或参考解。

新增CPU回归算例验证字符串/NaN结果与数值条件的首匹配优先规则。现共8项测试通过。execution_protocol增加environment_revision=2和明确兼容语义；测试payload包含兼容前缀，旧评分cache不会冒充新观测。

本修订发生在模型训练、生成和评分之前，只由参考程序与库源码解释驱动。重新对全部87个参考执行v2门禁；若任意失败仍停止，不删题。最终报告必须同时展示v1失败和v2状态，并将模型成绩标为自定义资源与NumPy select兼容协议的子集外测。

## 并行实现状态

新训练入口4项CPU测试及独立审查通过，未发现剩余Critical/Important静态问题；实际TRL检查点仍须GPU实跑验证。历史奖励审计入口6项CPU测试通过，真实日志分析待本提交后执行。PPO修复在独立源码快照完成，完整依赖CPU集成测试待部署。

此前两个agent的GitHub建分支/组合调用分别阻塞约1601/1892秒；已中断，任务切换为本地实现和主协调者统一提交，保留状态，不重复创建或重跑历史任务。
