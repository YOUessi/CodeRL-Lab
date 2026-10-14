from types import SimpleNamespace
from coderl_lab.analysis.exp7b_fence_recheck_resume import attributable_candidate_crash


def outcome(*, passed=False, result="runner-error", error=""):
    return SimpleNamespace(passed=passed, results=(result,), metadata={"error": error})


def test_recursive_python_stack_segfault_is_a_candidate_failure():
    fatal = ('Fatal Python error: Segmentation fault\n\n'
             'Current thread 0xabcdef (most recent call first):\n'
             '  File "<string>", line 46 in count_good_integers\n'
             '  File "<string>", line 46 in <genexpr>')
    assert attributable_candidate_crash(outcome(error=fatal))


def test_docker_or_harness_errors_must_fail_closed():
    fatal = ('Fatal Python error: Segmentation fault\n\n'
             'Current thread 0xabcdef\n  File "/lcb/testing_util.py" line 10')
    assert not attributable_candidate_crash(outcome(error=fatal))
    assert not attributable_candidate_crash(outcome(error="Cannot connect to Docker daemon"))
    assert not attributable_candidate_crash(outcome(error="Docker container exited 137"))
    assert not attributable_candidate_crash(outcome(result="invalid-runner-output", error=fatal))
    assert not attributable_candidate_crash(outcome(result="global-timeout", error=fatal))
    assert not attributable_candidate_crash(outcome(passed=True, error=fatal))
