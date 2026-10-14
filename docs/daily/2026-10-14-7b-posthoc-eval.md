# 2026-10-14 — EXP-7B：离线恢复、公共判分修复与官方隐藏测试

## 设备与现场
- Tang 的 Remote Desktop Commander 连接于 2026-10-14 12:37 CST 核实已恢复在线，GPU RTX 4090 Laptop 空闲显存约 15.5 GiB。
- 保留原始冻结 404 条候选：`artifacts/exp7b/n4/generation/checkpoint.jsonl`；SHA256 `6ac29a8fc1044526a9760b5e264ee7d8d8bf3fc46f773d46011f7228f11cda81`。
- 原始提交未经 Markdown 解析，404/404 均有前导 Python 代码围栏，公开/隐藏测试 0/404 属于**评测输入契约错误**，绝不是对7B编程能力的可靠估计。原始结果保留在 `results/exp7b/v4_n4_raw_official_summary.json`。
- 随后严格提取首个完整 Python Markdown 围栏（397 条归一化、7 条因围栏未闭合而保持原样），从原冻结候选出发，只有事后格式工程重判，不重新生成模型输出或调整 prompt、温度、top_p、max tokens。

## 本次实际故障复现与修复
- 接手时公开重判 `fence_recheck_v1` checkpoint 有 **278/404** 条，卡在 `abc370_g` 第3个候选 `sample_id=2` 的官方返回 `runner-error`。
- 重现两次同一公有测试、同一隔离 Docker 与固定1GB/6s限制，错误是：`Fatal Python error: Segmentation fault`，Python 提交代码 `File "<string>"` 中 `count_good_integers` 无限递归导致底层段错误；不是 Docker 服务/镜像故障。
- 没有修改或覆盖原版 `src/coderl_lab/analysis/exp7b_fence_recheck.py`，以保留已冻结的 `normalizer_module_sha256` 和候选身份；新建 `exp7b_fence_recheck_resume.py`，只在 `runner-error` 携带可验证的候选 Python 原生段错误栈特征时把该候选计作 **失败**。其他 Docker、官方执行器未知错误仍按原约定中断，不“跳过”失败。
- `tests/test_exp7b_fence_recheck.py` 与 `tests/test_exp7b_fence_recheck_resume.py` 在 Tang Python 环境实际运行：**6 passed**。
- 使用原始路径 `artifacts/exp7b/n4/fence_recheck_v1` 从278个已有 checkpoint继续，输入 identity SHA、normalized predictions SHA均匹配，保留旧 checkpoint。

## 404条公开测试事后完整结果
- **404/404** 官方公开用例重判完成（没有私有用例参与公开阶段）。
- **87/404** 个候选通过公开测试（21.53%）；至少一次公开通过的任务 **35/101**（34.65%）。
- 20,000次按题 Bootstrap (seed42) 的公开 solved@4 95% CI：[25.74%, 43.56%]。这一 CI 描述旧 v4 历史样本，不构成新分布验证。
- 候选级原生 Python segfault 数 **1**，其他已知资源上限/超时次数均为0。
- 机器摘要已上传 `results/exp7b/v4_n4_fence_recheck_public_summary.json`，GitHub提交 `ea331c028fb9cdc0f3f82255c1d79bc61d24468e`。

## 私有测试当前进度
- 仅在 404/404 公开检查和 `pre_private_frozen_inputs.json` SHA 冻结完毕后，才启动相同 404 条候选的官方隐藏测试。
- Tang 日志 `exp7b_fence_recheck_private.log`，本地 checkpoint `artifacts/exp7b/n4/fence_recheck_v1/private_checkpoint.jsonl`。
- 此文档创建时官方 private 判分仍未结束；**不能提前写私有正确率或 solved@4 数字**。
- 该分析始终标注 **posthoc historical v4 engineering recheck**，不能把事后修复当作新前瞻 holdout。

## 12:xx CST 官方 private 404/404 完整结束（事后工程复核）
- `exp7b_fence_recheck_resume.py` 沿用严格冻结的 404 条生成候选、397/404 条首个完整 Markdown Python代码块的统一剥离规则，不重新进行模型生成、也不改变 Docker 6秒/1GB、公有或私有测试。
- 在完整 **404/404** 公共测试冻结（87 个候选通过，35/101 题 solved@4）后，私有判分 `404/404` 顺利完成，`pre_private_frozen_inputs.json` 已在私有用例解码前落盘。没有覆盖原 raw 0/404 记录。
- 官方隐藏测试结果：**50/404 正确候选（12.38%）**，**20/101 题至少一次采样正确（19.80%）**。
- 按题 Bootstrap 20,000次 seed42 的隐藏 solved@4 95% CI **[12.87%, 27.72%]**，仅用于描述历史 v4 探索性结果，不是前瞻性统计成功。
- 执行器报告 candidate resource limit=4，global timeout=0，可确定由候选 Python 递归引发的 native segfault=1。没有未分类的官方执行器错误。
- 同一404个候选：公开通过87、隐藏通过50；其中**37** 个公开通过候选隐藏失败，**0** 个公开失败候选在合并隐藏测试下反而通过。题级公开 solved@4=35、隐藏 solved@4=20：15题至少一公开通过、没有隐藏通过。
- 难度分层隐藏 solved@4：Easy 13/22；Medium 7/34；Hard 0/45。平台分层：AtCoder 11/65；LeetCode 9/36。
- 与旧 EXP-004P 的同批101题对照（**仅描述性**）：旧 Qwen3-1.7B Base+SFT 贪心3/101、门控5/101；新 Qwen2.5-7B-Instruct 4次采样20/101。新7B成功20题中**18题是旧SFT贪心未解出**；旧SFT贪心3题中有1题新7B四采样仍未解出。**模型家族、Instruct/微调状态、prompt和采样预算不同，不能把差异归因为单一的参数规模因素。**
- 结果及源代码均已归档到 Github：
  - `results/exp7b/v4_n4_fence_recheck_public_summary.json`（commit `ea331c028fb9cdc0f3f82255c1d79bc61d24468e`）；
  - `results/exp7b/v4_n4_fence_recheck_private_summary.json`（commit `6c32aca9efb694542e43cdd2acb10e20e915b6f5`）；
  - `results/exp7b/v4_n4_posthoc_capacity_diagnostic.json`（commit `86388e672409c8923d8c587e9281dbdff0f53f8c`）；
  - 回归测试：`8 passed`；公开/私有结果对应相同候选SHA、101题ID、原样保存于Tang的 private checkpoint 和完整 source 文件不上传GitHub。
- 结论边界：这证明较强编码模型在旧 v4题集上四次采样**存在部分可达的正确程序**；之前raw零分是评测输入契约错误。仍须在**新独立时间/分布基准**与**同模板/同采样预算**模型对照上验证泛化，不能据此宣称局部低Margin干预在更强模型上有效。
