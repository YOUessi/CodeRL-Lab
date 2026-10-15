# EXP-019B：Qwen2.5-7B 的 MBPP+ 外部编程基准迁移验收

**2026-10-15，在从 MBPP+ 生成任何新模型输出和执行测试之前冻结。**

## 为什么开展
TRAIN-017 的 7B GRPO32 在新构造24道算法题×两随机种子上比同源7B SFT多10/192条正确候选，但四采样至少一解只净增1/48个种子-任务组合。该结果集中于柱状图等题型，现转入**不同来源公开 MBPP+**，检验能否迁移到真实自然语言编程函数问题，而不是继续修改已经看过的24道自制题。

**模型全部为相同 Qwen2.5-7B-Instruct 主模型**（`a09a35458c702b33eeacc393d103063234e8bc28`），同7B NF4 double quant BF16底座：
- 基线：TRAIN-011的7B SFT LoRA SHA256 `25eaf9d978f513e7235184c41fb0f7517333025675234112a9936edf93ba47b8`。
- 对照：TRAIN-017的7B GRPO32 LoRA SHA256 `47e0d8d6687efe1965c2ba7ff88f064b56fe3d10907c742824b116abe1ffc8b4`。
不使用1.5B结果代替7B，不重新训练模型，不根据本测试评分挑选Adapter。

## 数据和冻结选择
- 来源：公开 EvalPlus MBPP+ `evalplus/mbppplus` 的HF `test` parquet；来源文件保留Tang，原始 **378题**，内容SHA256 `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`，字节数1129135；官方 dataset card 表明378题。按文件SHA，而不是依赖可能变化的`main`标签定位版本。
- **结构预筛**：原始参考代码必须Python AST解析成功并且仅有一个顶层普通`def`，可以从其函数声明中取公开`entry_point`及函数签名；题干必须是非空字符串、≤600字符。只用题干/参考函数签名做公开模型提示，**不把函数体、参考代码、测试断言或预期结果给模型**。多顶层函数任务只按结构排除，记数；测试难易/内容不参与选题。
- 排序：在通过结构门禁的所有题中，按`SHA256('EXP019B|seed=42|'+str(task_id))`升序固定取**24道**，不因后来模型结果或超时重新选题。
- 输出：`public24.jsonl`（task_id/原prompt/要求的函数签名/模型输入消息），`private24_tests.jsonl`（仅task_id/entry_point/官方MBPP+增广断言测试），以及metadata manifest；GitHub只提交SHA和数量，原始代码、官方测试原文只保留Tang，不将私有test交给模型生成器。
- 官方测试字段是大型`test`脚本，包含外部导入/数值数组等，不等同于简单`test_list`；必须在已固定的 Docker EvalPlus 环境内执行。
- **此处评分采用自己冻结的MBPP+ test脚本沙箱执行，不是官方 EvalPlus CLI 的动态 per-task ground truth calibration/Base+Extra 拆分，也不能直接同官方排行榜比较。** 使用`semaremask-evalplus:latest`固定本地镜像ID `sha256:8734cd8b7868a1b0205384477cfdef64753ff979eb45d0249f60dc31a64c74ea`；非root、无网络、只读、drop all capabilities、单CPU、1GB、候选级最多15秒（实际执行所有增强测试），容器运行失败必须明确报错，不伪装模型不正确。记录reference黄金程序校验超时/失败数量，不按成绩剔除任务。
- 原始公开历史基准可能出现在7B预训练，属于**CodeRL-Lab首次独立来源迁移测试，而非时间前瞻未污染盲测**。

## 冻结生成方案
- 同24题，每题4条随机采样、SFT/GRPO32两臂，总计 **192条真实生成代码**；固定seed42+任务ID+sampleID导出的同一随机种子，不依赖模型分数动态选择。Qwen2.5官方chat模板，temperature0.8、top_p0.95、max_new_tokens256。三元输出必须完整保存原文+SHA并核对。
- 仅`generation_summary.json`确认192/192真实回答与SHA后，Docker scorer才有权限读取private24_tests/原始dataset测试字段。先冻结模型输出，再按相同官方断言逐条Docker判分；对比pass@1、四采样至少一次正确、候选正确率，并按**24题**配对20,000次Bootstrap，另报Token上限、代码语法/入口有效率和执行超时。
- 不把“每题四次至少一解”（observed solved@4）误称为大采样池无偏 pass@4 估计；区分这两种口径。

## 研究边界与工程规范
- 正确率变化是历史公开题24道的小样本探索，不证明真实大型软件项目的可推广提升。
- 如果评分存在环境/库冲突，记录gold reference和模型代码都失败的任务并报告，不得静默过滤或改选择规则。
- GitHub唯一事实源，Tang只进行GPU与Docker，所有源代码、固定配置、校验、日期日志、失败与修复及聚合结果按2026-10-15保存。
