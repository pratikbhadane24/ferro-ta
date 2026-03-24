from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


_ROOT = Path(__file__).resolve().parent.parent


def _run_cmd(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _read_toml(path: Path) -> dict[str, Any] | None:
    if tomllib is None or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except Exception:
        return None


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        return (
            _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
            or _run_cmd(["sysctl", "-n", "hw.model"])
            or platform.processor()
            or None
        )
    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            text = cpuinfo.read_text(encoding="utf-8", errors="ignore")
            for pattern in (r"model name\s+:\s+(.+)", r"Hardware\s+:\s+(.+)"):
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
        return platform.processor() or None
    if sys.platform.startswith("win"):
        return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or None
    return platform.processor() or None


def _total_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        raw = _run_cmd(["sysctl", "-n", "hw.memsize"])
        return int(raw) if raw and raw.isdigit() else None

    if sys.platform.startswith("linux"):
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            text = meminfo.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"MemTotal:\s+(\d+)\s+kB", text)
            if match:
                return int(match.group(1)) * 1024
        return None

    if sys.platform.startswith("win"):  # pragma: no cover
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys)
        except Exception:
            return None

    return None


def _cargo_release_profile() -> dict[str, Any] | None:
    cargo_toml = _read_toml(_ROOT / "Cargo.toml")
    if not cargo_toml:
        return None
    profile = cargo_toml.get("profile", {}).get("release")
    return profile if isinstance(profile, dict) else None


def git_info() -> dict[str, Any]:
    """Best-effort git metadata for reproducible benchmark artifacts."""
    return {
        "commit": _run_cmd(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run_cmd(["git", "status", "--porcelain"]) or ""),
        "branch": _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def runtime_info() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
    }


def build_info() -> dict[str, Any]:
    return {
        "rustc": _run_cmd(["rustc", "-Vv"]),
        "cargo": _run_cmd(["cargo", "-VV"]) or _run_cmd(["cargo", "-V"]),
        "cargo_release_profile": _cargo_release_profile(),
        "rustflags": os.environ.get("RUSTFLAGS"),
        "cargo_build_rustflags": os.environ.get("CARGO_BUILD_RUSTFLAGS"),
        "maturin_flags": os.environ.get("MATURIN_EXTRA_ARGS"),
    }


def package_versions(*names: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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
        "build": build_info(),
        "packages": package_versions("numpy", "ferro-ta"),
    }
    if fixtures:
        metadata["fixtures"] = [file_info(path) for path in fixtures]
    if extra:
        metadata.update(extra)
    return metadata
