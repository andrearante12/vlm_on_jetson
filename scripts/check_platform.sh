#!/usr/bin/env bash
# Pre-benchmark checklist: prints what every run will record, and warns about common problems.
set -uo pipefail
echo "== device";      tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
echo "== L4T";         head -1 /etc/nv_tegra_release 2>/dev/null
echo "== power mode";  nvpmodel -q 2>/dev/null | head -2
echo "== memory";      free -h | sed -n 1,3p
echo "== disk";        df -h . | tail -1
echo "== docker";      docker info 2>/dev/null | grep -iE "runtimes|default runtime" || echo "docker not available"
command -v tegrastats >/dev/null && echo "tegrastats: ok" || echo "WARNING: tegrastats not found (no power telemetry)"
nvpmodel -q 2>/dev/null | grep -q MAXN || echo "NOTE: not in a MAXN mode; for benchmark numbers run: sudo nvpmodel -m 0 && sudo jetson_clocks (check mode ids with nvpmodel -q --verbose)"
