from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _jsonable(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_official_run_test(lcb_repo: Path):
    module_path = lcb_repo / "lcb_runner" / "evaluation" / "testing_util.py"
    if not module_path.exists():
        raise RuntimeError(f"official testing_util.py missing: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "coderl_lab_pinned_lcb_testing_util",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_test


def main() -> None:
    payload = json.loads(sys.stdin.read())
    lcb_repo = Path("/lcb")
    if not lcb_repo.exists():
        raise RuntimeError("LiveCodeBench source mount /lcb is missing")

    run_test = _load_official_run_test(lcb_repo)

    sample = payload["sample"]
    code = str(payload["code"])
    timeout = int(payload.get("timeout", 6))

    results, metadata = run_test(
        sample,
        test=code,
        debug=False,
        timeout=timeout,
    )

    normalized = _jsonable(results)
    passed = bool(normalized) and all(x is True for x in normalized)

    print(
        json.dumps(
            {
                "passed": passed,
                "results": normalized,
                "metadata": _jsonable(metadata),
            },
            ensure_ascii=False,
            default=repr,
        )
    )


if __name__ == "__main__":
    main()
