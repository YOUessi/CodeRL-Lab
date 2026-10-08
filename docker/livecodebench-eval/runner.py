from __future__ import annotations

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


def main() -> None:
    payload = json.loads(sys.stdin.read())
    lcb_repo = Path("/lcb")
    if not lcb_repo.exists():
        raise RuntimeError("LiveCodeBench source mount /lcb is missing")

    sys.path.insert(0, str(lcb_repo))
    from lcb_runner.evaluation.testing_util import run_test

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
    bool_results = [bool(x) if isinstance(x, bool) else x for x in normalized]
    passed = bool(bool_results) and all(x is True for x in bool_results)

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
