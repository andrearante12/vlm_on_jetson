"""Snapshot of the hardware/software state a run was measured under.

Results are only comparable across Orin and Thor (and across power modes) if every run records
exactly what it ran on, so this is written to platform.json for every run.
"""

from __future__ import annotations

import datetime as dt
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip().strip("\x00")
    except OSError:
        return None


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _meminfo() -> dict:
    out = {}
    txt = _read("/proc/meminfo") or ""
    for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
        m = re.search(rf"^{key}:\s+(\d+) kB", txt, re.M)
        if m:
            out[key + "_MiB"] = int(m.group(1)) // 1024
    return out


def snapshot() -> dict:
    info: dict = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device_model": _read("/proc/device-tree/model"),
        "l4t_release": (_read("/etc/nv_tegra_release") or "").splitlines()[0:1],
        "nvpmodel": _run(["nvpmodel", "-q"]),
        "memory": _meminfo(),
    }
    # jetson_clocks --show needs root; record only whether we could read the GPU max/cur freq.
    for p in Path("/sys/devices/platform").glob("*gpu*/devfreq/*/cur_freq"):
        info["gpu_cur_freq_hz"] = _read(str(p))
        info["gpu_max_freq_hz"] = _read(str(p.parent / "max_freq"))
        break
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # torch not installed (e.g. server-backend-only client)
        info["torch"] = None
    for mod in ("transformers", "tensorrt", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    rev = _run(["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "--short", "HEAD"])
    info["git_rev"] = rev or None
    return info
