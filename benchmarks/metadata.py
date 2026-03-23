from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_info() -> dict[str, Any]:
    """Best-effort git metadata for reproducible benchmark artifacts."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = None

    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        dirty = None

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        branch = None

    return {"commit": commit, "dirty": dirty, "branch": branch}


def runtime_info() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }


def file_info(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    data = file_path.read_bytes()
    return {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def benchmark_metadata(
    suite: str,
    *,
    fixtures: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "suite": suite,
        "runtime": runtime_info(),
        "git": git_info(),
    }
    if fixtures:
        metadata["fixtures"] = [file_info(path) for path in fixtures]
    if extra:
        metadata.update(extra)
    return metadata
