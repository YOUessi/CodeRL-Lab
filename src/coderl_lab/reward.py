from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TrainingReward:
    syntax: float
    public_pass_rate: float
    all_public_pass_bonus: float
    total: float


@dataclass(frozen=True)
class DependencyAnalysis:
    syntax_ok: bool
    entry_point_found: bool
    unresolved_names: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            self.syntax_ok
            and self.entry_point_found
            and not self.unresolved_names
        )


@dataclass(frozen=True)
class ExecutionStageReward:
    syntax: float
    dependency_complete: float
    runtime_clean_rate: float
    public_pass_rate: float
    all_public_pass_bonus: float
    total: float


def compute_training_reward(
    *,
    syntax_ok: bool,
    public_pass_rate: float,
    syntax_weight: float = 0.10,
    public_test_weight: float = 0.80,
    all_public_pass_bonus_weight: float = 0.10,
) -> TrainingReward:
    """Compute the original outcome-style reward.

    Hidden tests are intentionally excluded. This function is kept unchanged
    in semantics so historical EXP-003 / EXP-006A baselines remain reproducible.
    """
    if not 0.0 <= public_pass_rate <= 1.0:
        raise ValueError("public_pass_rate must be in [0, 1]")

    weights = syntax_weight + public_test_weight + all_public_pass_bonus_weight
    if abs(weights - 1.0) > 1e-9:
        raise ValueError("reward weights must sum to 1.0")

    syntax = 1.0 if syntax_ok else 0.0
    all_public = 1.0 if syntax_ok and public_pass_rate == 1.0 else 0.0
    total = (
        syntax_weight * syntax
        + public_test_weight * public_pass_rate
        + all_public_pass_bonus_weight * all_public
    )
    return TrainingReward(
        syntax=syntax,
        public_pass_rate=public_pass_rate,
        all_public_pass_bonus=all_public,
        total=total,
    )


def _module_defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if isinstance(node, ast.Import):
                    names.add(alias.asname or alias.name.split(".")[0])
                else:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            else:
                targets = [node.target]
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
    return names


def _function_local_names(fn: ast.AST) -> set[str]:
    names: set[str] = set()

    args = getattr(fn, "args", None)
    if args is not None:
        for arg in (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
        ):
            names.add(arg.arg)
        if args.vararg is not None:
            names.add(args.vararg.arg)
        if args.kwarg is not None:
            names.add(args.kwarg.arg)

    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if isinstance(node, ast.Import):
                    names.add(alias.asname or alias.name.split(".")[0])
                else:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            names.add(node.name)
    return names


def _function_loaded_names(fn: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def analyze_dependency_completeness(
    code: str,
    *,
    entry_point: str,
    setup_code: str = "",
) -> DependencyAnalysis:
    """Find obvious unresolved runtime names in the target function.

    This is deliberately conservative and fully static. It treats builtins,
    module-level imports/definitions and setup-code definitions as available.
    It does not use hidden tests and never executes the candidate.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return DependencyAnalysis(
            syntax_ok=False,
            entry_point_found=False,
            unresolved_names=(),
        )

    setup_names: set[str] = set()
    if setup_code.strip():
        try:
            setup_tree = ast.parse(setup_code)
        except SyntaxError:
            setup_tree = ast.Module(body=[], type_ignores=[])
        setup_names = _module_defined_names(setup_tree)

    target: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == entry_point:
                target = node
                break

    if target is None:
        return DependencyAnalysis(
            syntax_ok=True,
            entry_point_found=False,
            unresolved_names=(),
        )

    available = (
        set(dir(builtins))
        | _module_defined_names(tree)
        | setup_names
        | _function_local_names(target)
    )
    unresolved = tuple(
        sorted(_function_loaded_names(target) - available)
    )
    return DependencyAnalysis(
        syntax_ok=True,
        entry_point_found=True,
        unresolved_names=unresolved,
    )


def case_runtime_clean(case: Any) -> bool:
    """Whether a public test reached a logical verdict without runtime failure.

    For assertion-style MBPP tests, a wrong answer raises AssertionError and is
    still a clean execution. NameError/TypeError/etc. and timeout are runtime
    failures. Call-style tests with a wrong returned value have error=None and
    are also considered clean.
    """
    if isinstance(case, dict):
        timed_out = bool(case.get("timed_out", False))
        error = case.get("error")
    else:
        timed_out = bool(getattr(case, "timed_out", False))
        error = getattr(case, "error", None)

    if timed_out:
        return False
    if error is None:
        return True
    return "AssertionError" in str(error)


def compute_runtime_clean_rate(cases: Iterable[Any]) -> float:
    case_list = list(cases)
    if not case_list:
        return 0.0
    return sum(case_runtime_clean(case) for case in case_list) / len(case_list)


def compute_execution_stage_reward(
    *,
    syntax_ok: bool,
    dependency_complete: bool,
    runtime_clean_rate: float,
    public_pass_rate: float,
    syntax_weight: float = 0.05,
    dependency_weight: float = 0.10,
    runtime_clean_weight: float = 0.10,
    public_test_weight: float = 0.65,
    all_public_pass_bonus_weight: float = 0.10,
) -> ExecutionStageReward:
    """Dense, fully verifiable execution-stage reward for EXP-004A."""
    for name, value in {
        "runtime_clean_rate": runtime_clean_rate,
        "public_pass_rate": public_pass_rate,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    weights = (
        syntax_weight
        + dependency_weight
        + runtime_clean_weight
        + public_test_weight
        + all_public_pass_bonus_weight
    )
    if abs(weights - 1.0) > 1e-9:
        raise ValueError("execution-stage reward weights must sum to 1.0")

    syntax = 1.0 if syntax_ok else 0.0
    dependency = 1.0 if dependency_complete else 0.0
    all_public = 1.0 if syntax_ok and public_pass_rate == 1.0 else 0.0

    total = (
        syntax_weight * syntax
        + dependency_weight * dependency
        + runtime_clean_weight * runtime_clean_rate
        + public_test_weight * public_pass_rate
        + all_public_pass_bonus_weight * all_public
    )
    return ExecutionStageReward(
        syntax=syntax,
        dependency_complete=dependency,
        runtime_clean_rate=runtime_clean_rate,
        public_pass_rate=public_pass_rate,
        all_public_pass_bonus=all_public,
        total=total,
    )
