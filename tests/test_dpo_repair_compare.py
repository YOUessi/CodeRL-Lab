from coderl_lab.analysis.dpo_repair_compare import compare


def eval_summary(p1, p4, p8, p16, hidden, solved, syntax):
    return {
        "pass_at_k": {
            "pass@1": p1,
            "pass@4": p4,
            "pass@8": p8,
            "pass@16": p16,
        },
        "hidden_mean_pass_rate": hidden,
        "solved_tasks": solved,
        "all_samples_correct_tasks": 1,
        "syntax_failures": syntax,
    }


def diag(dep, runtime, nameerr, re_count, math_count):
    return {
        "dependency_incomplete_count": dep,
        "runtime_unclean_count": runtime,
        "top_unresolved_names": [["re", re_count], ["math", math_count]],
        "runtime_failure_types": [["NameError", nameerr]],
    }


def test_compare_repair_metrics() -> None:
    out = compare(
        sft_eval=eval_summary(0.3, 0.4, 0.5, 0.6, 0.3, 5, 4),
        dpo_eval=eval_summary(0.31, 0.41, 0.52, 0.61, 0.32, 6, 2),
        sft_diag=diag(20, 30, 15, 10, 5),
        dpo_diag=diag(12, 20, 8, 4, 2),
        passk_bootstrap={"pass@1": {"observed_delta": 0.01}},
    )
    assert out["validation"]["delta"]["pass@1"] == 0.010000000000000009
    assert out["diagnostics"]["delta"]["dependency_incomplete_count"] == -8
    assert out["diagnostics"]["delta"]["unresolved_re"] == -6
