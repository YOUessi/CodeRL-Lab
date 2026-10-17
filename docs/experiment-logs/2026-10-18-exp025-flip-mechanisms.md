# 2026-10-18｜EXP-025 正确性翻转与依赖错误机制审计

按新冻结MBPP+96题、固定采样索引，从Tang Docker判题日志直接比较“严格奖励 - 旧完整奖励”，不执行候选代码，也不公开原始答案或隐藏测试。

- Seed43：3修复（MBPPPlus/409 样本1 NameError→通过；/556 样本0与1 AssertionError→通过）、4回退（/137样本1 AssertionError；/432样本2 TypeError；/732样本3 AssertionError；/757样本2 NameError）。
- Seed44：6修复（/12样本1 TypeError、/124样本0 TypeError、/137样本1/2 AssertionError、/556样本0 AssertionError、/95样本1 NameError →通过）、3回退（/297样本1 TypeError；/556样本1 AssertionError；/732样本3 AssertionError）。
- 合计跨两种子16次配对正确性翻转（9修复、7回退），涉及两个随机训练种子，与全部任务总体效果不能独立混同。
- NameError：seed43各有一次修复与一次新引入，净0；seed44修复一次且未出现由正确变NameError，净-1。与总NameError次数 seed43两组均24、seed44 旧25/严格24一致。
- 稳定共现的题目例如/732样本3在两个种子均由旧奖励正确→严格奖励错误；/556样本0在两个种子都出现旧奖励错误→严格奖励正确。这些是针对性机制审计候选，但选择自已观测到的翻转，不能当做独立的泛化估计。

核心结论：新严格奖励尚未稳定减少NameError或提升正确率。应优先审计/732和/556的程序变化及训练奖励提供的依赖信号，再预注册独立跨数据域验证。
