"""Background power/utilisation sampling via tegrastats.

Inside a container tegrastats must be bind-mounted from the host (see docker/compose.yaml). When it
is unavailable we fall back to /proc/meminfo so memory is still recorded.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
_CPU = re.compile(r"CPU \[([^\]]*)\]")
_TEMP = re.compile(r"\b(cpu|gpu|tj|soc\d?)@(-?[\d.]+)C", re.I)
_RAIL = re.compile(r"\b(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+) (\d+)mW/(\d+)mW")

FIELDS = ["t", "ram_used_mb", "ram_total_mb", "swap_used_mb", "gpu_util_pct", "cpu_util_pct",
          "power_total_mw", "temp_cpu_c", "temp_gpu_c", "temp_tj_c", "raw"]


def parse_tegrastats_line(line: str) -> dict:
    row: dict = {}
    if m := _RAM.search(line):
        row["ram_used_mb"], row["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    if m := _SWAP.search(line):
        row["swap_used_mb"] = int(m.group(1))
    if m := _GR3D.search(line):
        row["gpu_util_pct"] = int(m.group(1))
    if m := _CPU.search(line):
        loads = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
        if loads:
            row["cpu_util_pct"] = round(sum(loads) / len(loads), 1)
    for name, val in _TEMP.findall(line):
        key = {"cpu": "temp_cpu_c", "gpu": "temp_gpu_c", "tj": "temp_tj_c"}.get(name.lower())
        if key:
            row[key] = float(val)
    rails = {name: int(cur) for name, cur, _avg in _RAIL.findall(line)}
    # Total board input power: VDD_IN on Orin; Thor/other modules may name it differently.
    for name in ("VDD_IN", "VIN_SYS_5V0", "VDD_SYS_5V0"):
        if name in rails:
            row["power_total_mw"] = rails[name]
            break
    else:
        if rails:
            row["power_total_mw"] = sum(rails.values())
    return row


class Telemetry:
    """Context manager that samples tegrastats into a CSV for the duration of a run."""

    def __init__(self, out_csv: str | Path, interval_ms: int = 500, enabled: bool = True):
        self.out_csv = Path(out_csv)
        self.interval_ms = interval_ms
        self.enabled = enabled
        self.rows: list[dict] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.source = None

    def __enter__(self):
        if not self.enabled:
            return self
        exe = shutil.which("tegrastats") or ("/usr/bin/tegrastats" if Path("/usr/bin/tegrastats").exists() else None)
        if exe:
            try:
                self._proc = subprocess.Popen([exe, "--interval", str(self.interval_ms)],
                                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                self.source = "tegrastats"
                self._thread = threading.Thread(target=self._read_tegrastats, daemon=True)
            except OSError:
                self._proc = None
        if self._proc is None:
            self.source = "meminfo"
            self._thread = threading.Thread(target=self._poll_meminfo, daemon=True)
        self._thread.start()
        return self

    def _read_tegrastats(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            row = parse_tegrastats_line(line)
            row["t"] = time.time()
            row["raw"] = line.strip()
            self.rows.append(row)

    def _poll_meminfo(self):
        while not self._stop.wait(self.interval_ms / 1000):
            try:
                txt = Path("/proc/meminfo").read_text()
            except OSError:
                continue
            total = int(re.search(r"MemTotal:\s+(\d+)", txt).group(1)) // 1024
            avail = int(re.search(r"MemAvailable:\s+(\d+)", txt).group(1)) // 1024
            self.rows.append({"t": time.time(), "ram_used_mb": total - avail, "ram_total_mb": total})

    def __exit__(self, *exc):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)
        if self.enabled:
            self.out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)
        return False

    def summary(self, t0: float | None = None, t1: float | None = None) -> dict:
        rows = [r for r in self.rows if (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] <= t1)]
        out: dict = {"telemetry_source": self.source, "telemetry_samples": len(rows)}
        for key in ("ram_used_mb", "gpu_util_pct", "power_total_mw", "temp_tj_c", "temp_gpu_c"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                out[f"{key}_mean"] = round(sum(vals) / len(vals), 1)
                out[f"{key}_max"] = max(vals)
        if "power_total_mw_mean" in out and rows:
            dur = rows[-1]["t"] - rows[0]["t"]
            out["energy_j"] = round(out["power_total_mw_mean"] / 1000 * dur, 2)
        return out
