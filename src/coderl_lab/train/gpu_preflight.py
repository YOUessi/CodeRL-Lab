"""Protect the single Tang GPU without mistakenly blocking known display daemons.

Never terminate another process. Report exactly which process blocks training.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import PurePath
from typing import Any

# Read-only GPU compute process queries may include ToDesk on Tang even while
# no model training is running. Everything else is conservatively blocked.
DISPLAY_ONLY = frozenset({"ToDesk_Session", "Xorg", "Xwayland", "gnome-shell", "kwin_x11"})


def classify_gpu_compute_processes(raw_csv: str) -> dict[str, Any]:
    blocked: list[dict[str, str]] = []
    allowed_display: list[dict[str, str]] = []
    for row in csv.reader(io.StringIO(raw_csv)):
        if not row or not "".join(row).strip():
            continue
        if len(row) != 3:
            if len(row) == 1 and row[0].strip().lower() in (
                "no running processes found", "[not supported]", "n/a"
            ):
                continue
            raise ValueError(f"unparseable GPU process line with {len(row)} columns")
        pid, path, gpu_mem = (part.strip() for part in row)
        if not pid.isdigit() or not path:
            raise ValueError("bad GPU process PID or command name")
        name = PurePath(path).name
        record = {"pid": pid, "process": name, "gpu_memory": gpu_mem}
        (allowed_display if name in DISPLAY_ONLY else blocked).append(record)
    return {
        "safe_for_single_model_job": not blocked,
        "allowed_display_daemons": allowed_display,
        "blocking_processes": blocked,
    }


def assert_gpu_exclusive(*, min_free_mib: int) -> dict[str, Any]:
    if min_free_mib < 1:
        raise ValueError("invalid free GPU memory requirement")
    compute = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=True,
    )
    result = classify_gpu_compute_processes(compute.stdout)
    if result["blocking_processes"]:
        raise RuntimeError(
            "Tang GPU occupied; no training started: "
            + ", ".join(
                f"pid={x['pid']} process={x['process']}"
                for x in result["blocking_processes"]
            )
        )
    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    try:
        free = [int(line.strip()) for line in mem.stdout.splitlines() if line.strip()]
    except ValueError as exc:
        raise RuntimeError("cannot parse GPU free memory") from exc
    if len(free) != 1 or free[0] < min_free_mib:
        raise RuntimeError(
            f"not enough free GPU RAM for single model: {free} MiB, "
            f"need at least {min_free_mib} MiB"
        )
    result["free_memory_mib"] = free[0]
    result["min_required_mib"] = min_free_mib
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-mib", type=int, default=6000)
    args = parser.parse_args()
    result = assert_gpu_exclusive(min_free_mib=args.min_free_mib)
    print(
        f"GPU preflight passed: {result['free_memory_mib']} MiB free; "
        f"no non-display compute process active"
    )


if __name__ == "__main__":
    main()
