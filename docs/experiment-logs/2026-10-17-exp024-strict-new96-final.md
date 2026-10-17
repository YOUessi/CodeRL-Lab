# 2026-10-17｜EXP-024 strict reward completed two-seed external new96 audit

## Ground truth (Tang)

Both strict reward 7B training runs completed 32 optimizer steps and all MBPP+ new96 generated 768 paired candidate outputs per seed, with 768 Docker-scored records per seed. New96 cohort is nonoverlapping with previously tested 24+96 items, from frozen MBPP+ SHA256 dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b.

| Metric | SFT common baseline | strict seed43 | strict seed44 |
|---|---:|---:|---:|
| Correct candidates /384 |245|244|247|
| Fixed first sample success /96 |58|59|60|
| Observed solved at four /96 |73|72|74|
| NameError candidate verdict count |24|24|24|
| Net candidate accuracy delta (pp) |—|-0.2604|+0.5208|
| Paired task bootstrap 95% interval (pp) |—|[-1.8229,+1.3021]|[-0.7813,+1.8229]|
| Reference golden controls passed |96|96|96|

Training strict seed43: changed adapter parameter elements 1,852,469; reward-variation groups 28; final adapter SHA d17670683ae80cb14a8a9f60fd1cc81fc71c768fa2592a878e3d4ea81079b1b3. Seed44: changed elements 1,847,678; variation groups 30; final adapter SHA f3a909a17c548bfeaac468d65d9a3c99d4bc44b39a1b629b2a41a90adca92564.

## Conclusions

Both seeds produced actual GRPO updates. External candidate correctness changes have opposing signs and both CIs include 0; 4-sample coverage also changes in opposite directions. Number of NameError verdicts stays 24/384 in both strict arms, same as SFT; no observed aggregate reduction. These are **not** established improvements in programming generalization.

The essential next contrast is to evaluate prior full-reward GRPO seeds 43 and 44 on *the identical frozen new96 cohort*, with exactly the same prompts, model revision, sampling indices and Docker test suite. Without this, strict-vs-old reward improvements cannot be claimed. For leakage control generation must finish and be SHA-frozen before private tests are read. Distinguish new-task-in-same-MBPP+ benchmark from genuine cross-benchmark transfer; consider HumanEval+ or time-controlled LiveCodeBench thereafter.
