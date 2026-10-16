#!/usr/bin/env bash
# Run a trails command in the container that matches this Jetson.
#   docker/run.sh info
#   docker/run.sh run configs/experiments/baseline.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "")
case "$model" in
  *Thor*) profile=thor ;;
  *Orin*) profile=orin ;;
  *) echo "unrecognised device '$model'; set PROFILE=orin|thor" >&2; profile=${PROFILE:?} ;;
esac
profile=${PROFILE:-$profile}
exec docker compose -f docker/compose.yaml --profile "$profile" run --rm "trails-$profile" "$@"
