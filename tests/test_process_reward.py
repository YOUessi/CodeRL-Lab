from types import SimpleNamespace

import pytest

from coderl_lab.reward import (
    analyze_dependency_completeness,
    case_runtime_clean,
    compute_execution_stage_reward,
    compute_runtime_clean_rate,
    compute_training_reward,
)


def test_missing_re_import_is_detected() -> None:
    code = """
def search_literal(pattern, text):
    result = re.search(pattern, text)
    return result.span() if result else (-1, -1)
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="search_literal",
    )
    assert result.syntax_ok
    assert result.entry_point_found
    assert result.unresolved_names == ("re",)
    assert not result.complete


def test_re_import_restores_dependency_completeness() -> None:
    code = """
import re

def search_literal(pattern, text):
    result = re.search(pattern, text)
    return result.span() if result else (-1, -1)
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="search_literal",
    )
    assert result.complete
    assert result.unresolved_names == ()


def test_setup_code_can_supply_dependency() -> None:
    code = """
def root(x):
    return sqrt(x)
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="root",
        setup_code="from math import sqrt",
    )
    assert result.complete


def test_module_helper_is_available() -> None:
    code = """
def helper(x):
    return x + 1

def solve(x):
    return helper(x)
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="solve",
    )
    assert result.complete


def test_local_variables_are_not_false_dependencies() -> None:
    code = """
def solve(xs):
    total = 0
    for x in xs:
        total += x
    return total
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="solve",
    )
    assert result.complete


def test_runtime_clean_distinguishes_logical_failure_from_crash() -> None:
    passed = SimpleNamespace(error=None, timed_out=False)
    wrong_answer = SimpleNamespace(
        error="Traceback...\nAssertionError\n",
        timed_out=False,
    )
    name_error = SimpleNamespace(
        error="Traceback...\nNameError: name 're' is not defined\n",
        timed_out=False,
    )
    timeout = SimpleNamespace(error="execution timed out", timed_out=True)

    assert case_runtime_clean(passed)
    assert case_runtime_clean(wrong_answer)
    assert not case_runtime_clean(name_error)
    assert not case_runtime_clean(timeout)

    rate = compute_runtime_clean_rate(
        [passed, wrong_answer, name_error, timeout]
    )
    assert rate == pytest.approx(0.5)


def test_execution_stage_reward_weights() -> None:
    reward = compute_execution_stage_reward(
        syntax_ok=True,
        dependency_complete=True,
        runtime_clean_rate=1.0,
        public_pass_rate=0.0,
    )
    assert reward.total == pytest.approx(0.25)

    correct = compute_execution_stage_reward(
        syntax_ok=True,
        dependency_complete=True,
        runtime_clean_rate=1.0,
        public_pass_rate=1.0,
    )
    assert correct.total == pytest.approx(1.0)


def test_original_outcome_reward_is_unchanged() -> None:
    reward = compute_training_reward(
        syntax_ok=True,
        public_pass_rate=0.5,
    )
    assert reward.total == pytest.approx(0.5)


def test_lambda_parameters_are_not_false_dependencies() -> None:
    code = """
def largest_pos(values):
    return max(filter(lambda x: x > 0, values))
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="largest_pos",
    )
    assert result.complete
    assert result.unresolved_names == ()


def test_nested_recursive_helper_is_not_false_dependency() -> None:
    code = """
def get_lcm(values):
    def gcd(a, b):
        if a == 0:
            return b
        return gcd(b % a, a)
    result = values[0]
    for value in values[1:]:
        result = result * value // gcd(result, value)
    return result
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="get_lcm",
    )
    assert result.complete
    assert result.unresolved_names == ()


def test_comprehension_target_is_not_false_dependency() -> None:
    code = """
def squares(values):
    return [x * x for x in values]
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="squares",
    )
    assert result.complete
    assert result.unresolved_names == ()


def test_missing_global_inside_lambda_is_detected() -> None:
    code = """
def roots(values):
    return list(map(lambda x: math.sqrt(x), values))
"""
    result = analyze_dependency_completeness(
        code,
        entry_point="roots",
    )
    assert result.unresolved_names == ("math",)
    assert not result.complete
