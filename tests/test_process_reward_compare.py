from coderl_lab.analysis.process_reward_compare import compare


def dyn(zero_grad, zero_std, public, entropy, dep=None, clean=None):
    return {
        "zero_grad_fraction": zero_grad,
        "overall_mean": {
            "frac_reward_zero_std": zero_std,
            "reward/public_mean": public,
            "reward/dependency_complete_mean": dep,
            "reward/runtime_clean_mean": clean,
            "entropy": entropy,
        },
    }


def ev(p1, p4, p8, p16, hidden, solved, syntax):
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


def diag(dep, runtime):
    return {
        "dependency_incomplete_count": dep,
        "runtime_unclean_count": runtime,
        "top_unresolved_names": [],
        "runtime_failure_types": [],
    }


def test_compare_process_reward() -> None:
    out = compare(
        baseline_dynamics=dyn(0.3, 0.6, 0.4, 0.2),
        process_dynamics=dyn(0.2, 0.4, 0.45, 0.21, 0.8, 0.75),
        baseline_eval=ev(0.3, 0.5, 0.6, 0.7, 0.35, 60, 10),
        process_eval=ev(0.32, 0.52, 0.62, 0.72, 0.38, 62, 4),
        baseline_diag=diag(200, 300),
        process_diag=diag(120, 180),
    )
    assert out["training"]["zero_grad_fraction_delta"] == -0.1
    assert out["validation"]["delta"]["pass@16"] == 0.02
    assert out["diagnostics"]["delta"]["dependency_incomplete_count"] == -80
