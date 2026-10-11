# EXP-004P: Prospective capability-conditioned cross-distribution transfer

Date: 2026-10-11
Status: protocol frozen in GitHub; **no new GPU generations, hidden evaluations, or performance results**.
Parent: `exp/exp004o-transfer-boundaries`.
Prior outcomes (development evidence only): MBPP K +8/500; LCB v6 L +1/175; LCB v5 N +0/167.

## Scientific question
When a public-test verifier flags an incorrect baseline, does local low-margin trajectory perturbation fail to rescue it because the policy has almost no *correct program support* on that task, rather than because perturbation fails to alter trajectories?

This is a new **prospective** hypothesis. The 2026-10-09 EXP-004O retrospective analysis is not causal identification.

## Non-negotiable data separation
1. Never use LCB v5 or v6 private labels as the primary experiment set or for selecting perturbation thresholds, decoding lengths or models.
2. Select a new, non-overlapping official competitive-programming release slice (or otherwise a newly curated and independently sourced set), **before inspecting labels**, using question IDs + normalized question/starter-code hash to exclude all LCB v5/v6 tasks. Record publication/contest dates and model release date; do not assert contamination-free evaluation.
3. At generation time expose only public statements, starter code, supplied public tests and public metadata. Private tests are decoded only once all candidate outputs, checkpoint hashes, selection rules and success criteria are frozen.
4. No task exclusion based on difficulty, output correctness, parse failure, runtime exception, or model failure. If an external dataset has unusable rows, declare deterministic schema-based exclusions and counts before any inference.
5. Never feed private tests or their failures into prompt, reward, gate, scoring of candidates, or subsequent trials.

## Before-GPU data gate
- Manifest: upstream repository, exact revision, raw SHA256, public-view SHA256, task IDs and hashes, issue/contest dates, task count, platform and difficulty.
- Disjointness audit: compare with v5 and v6 IDs and canonicalized problem bodies. Zero overlaps or deterministic exclusion; report all exclusions.
- Verify official evaluator pinned commit; container time/memory limits, file I/O policy, and stdin versus functional checker formats.
- Freeze model checkpoints and the SFT LoRA adapter hashes; run CPU tests on representative public-only fixtures.
- If this gate fails, record failure and **do not start GPU or inspect private outcomes**.

## Arms (paired tasks; same model checkpoint, prompt, seeds, and checker)
A. SFT greedy 512 (original baseline).
B. SFT gated-low with frozen EXP-004J/K intervention: public fail AND any low-margin token among first 128; low threshold <=0.05; reference token logit bias -0.25; no alteration elsewhere.
C. SFT gated-high control with threshold >=0.20, otherwise matched to B.
D. SFT always-low (no verifier gate) with the same low-margin rule.
E. SFT multi-sample baseline n=16 (predeclared fixed seeds; **secondary**, no changing B after seeing E).
F. Optional independent stronger **preselected** coding policy (different capacity) with its own *paired* no-intervention/gated-low comparison only if weights, revision, and prompt template can be declared **before** any private outcomes. Do not compare its raw accuracy with 1.7B as a causal effect of capacity.

Do not change max_new_tokens, bias, threshold, public test budget, or prompt using v5/v6 observed hidden scores. If n=16 cannot run under available resources, mark secondary E missing, not silently lower n or replace it post-hoc.

## Primary endpoint and analysis
- Primary effect: mean paired hidden-correctness difference B − A across **all** preregistered tasks.
- Paired task bootstrap 20,000 resamples, seed 42; report 95% percentile CI and discordant counts (wrong-to-correct, correct-to-wrong).
- Confirmatory positive replication only if two-sided 95% CI lower bound > 0. Otherwise report non-replication, including a zero effect.
- Report the full cohort, not only gate-selected tasks; do not selectively present easier subsets as confirmatory.

## Mechanism endpoints (secondary/descriptive)
- Gate coverage, public-pass protection, post-hoc gate precision/recall.
- Among gate-triggered tasks, exact trajectory changed fraction and wrong→correct fraction; **distinguish these denominators**.
- From E, task-level oracle success@16 and conditional rescue rates by n=16 solvability. This conditioning is exploratory because success@16 is an observed outcome, not an exogenous task attribute.
- Completion token cap, parsed AST, runtime/timeout/resource-limit counts per arm.
- Paired B−C and B−D, plus difficulty/platform summaries clearly marked exploratory. No more effect-significance claims from slicing to convenient subsets.
- Show uncertainty (Wilson intervals) for rare rescue fractions, not only proportions.
- Release aggregated public-only metadata, hashes, evaluator revision, commands, seed and environment. Do not commit private test contents or sensitive model artifacts.

## Falsifiable interpretation
- If B changes many trajectories but rescue is near zero while E also has almost no successes, this is consistent with a limited accessible-solution hypothesis, **not proof of it**.
- If E frequently solves a task but B rarely rescues it, the bottleneck-choice / intervention policy is implicated.
- If both B and C rescue at comparable rates, the low-margin-specific causal story weakens.
- If B and D perform similarly, verifier-gate necessity does not replicate.
- Only independent, prospective data can establish new external-generalization evidence.

## Execution checkpoints
0. Save preregistration commit and input manifest SHA.
1. Run CPU preprocessing and regression tests; archive counts and pass/fail log in `docs/daily/2026-10-11.md`.
2. Tang GPU: capture `nvidia-smi` / available VRAM; run memory preflight; execute generation with checkpoint/resume and frozen configuration.
3. Validate complete row count + determinism + frozen runner hashes **before** invoking official private checker.
4. Evaluate official hidden tests once; publish machine-readable aggregate summary and failure categories; never re-tune on this private set.
5. Perform independent post-hoc mechanism audit and commit positive **or negative** outcome.

## Resource and stop conditions
GPU work uses Tang only once needed. Stop and record inability to run if external GPU compute is busy, checkpoint SHA mismatches, public/private boundary is violated, or the official evaluator is unavailable. No invented outcomes or implied asynchronous execution.


## 2026-10-11: actual public-data source check / protocol amendment (before any new outputs)
- The currently inspected official `code_generation_lite` source exposes fine-grained `test.jsonl` through `test6.jsonl`; **no available v7 slice was confirmed**. Do not fabricate one.
- A historical v4-only cohort (`test4.jsonl`) is chosen as the reproducible **independent cohort feasibility track**, compared with used v5/v6 IDs and normalized statement hashes at pinned source revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.
- **Important protocol correction:** v4 predates v5/v6. It is not a prospectively later or contamination-free temporal holdout. Any v4 hidden evaluation must be labeled *historical independent-cohort exploratory replication*, not the original prospectively newer-distribution confirmatory primary study.
- Original prospective post-v6 primary remains **blocked** until a verifiably newer independent benchmark with executable public/private evaluation is identified and pinned. Do not replace that primary with v4 silently.
- New public-only code: `src/coderl_lab/datasets/exp004p_public_v4.py` and `tests/test_exp004p_public_v4.py`. Their existence does not establish that all raw data were downloaded or the unit tests passed.
- No new generation or private outcomes have been produced as part of this amendment.
