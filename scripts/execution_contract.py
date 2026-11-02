"""Enforce the local-build / remote-run split for MegaMoE npusim."""
from __future__ import annotations

import platform
import sys
from pathlib import Path


LOCAL_BUILD_PYTHON = Path("/home/t00826093/miniforge3/envs/tang").resolve()
REMOTE_RUN_PYTHON = Path("/home/t00826093/miniconda3/envs/tang").resolve()


def _require_python(expected: Path, stage: str, allowed_location: str) -> None:
    actual = Path(sys.prefix).resolve()
    if actual != expected:
        raise RuntimeError(
            f"{stage} must run on the {allowed_location} using {expected}; "
            f"current Python is {actual}. Source checkout, instrumentation changes, "
            "case generation and compilation are local-only; model1 is for npusim execution only."
        )


def require_local_build(stage: str) -> None:
    _require_python(LOCAL_BUILD_PYTHON, stage, "local build machine")


def require_remote_simulation(stage: str) -> None:
    _require_python(REMOTE_RUN_PYTHON, stage, "model1 simulation machine")


def execution_evidence(stage: str, location: str) -> dict[str, str]:
    return {
        "stage": stage,
        "execution_location": location,
        "python": sys.executable,
        "python_prefix": sys.prefix,
        "machine": platform.node(),
    }
