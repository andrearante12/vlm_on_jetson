#!/usr/bin/env bash
# Copy footage, ground truth and (optionally) past results from this machine to another Jetson.
# Code travels through git; data does not (videos and frame-embedded viewer pages stay out of the repo).
#
#   scripts/transfer_data.sh user@thor-host [/path/to/vlm_on_jetson]   # videos + GT + annotations
#   WITH_RESULTS=1 scripts/transfer_data.sh user@thor-host              # ...plus results/ (run pages)
set -euo pipefail
dest=${1:?usage: transfer_data.sh user@host [remote_repo_dir]}
remote_dir=${2:-~/vlm_on_jetson}
cd "$(dirname "$0")/.."
ssh "$dest" "mkdir -p $remote_dir/data/videos $remote_dir/data/gt $remote_dir/data/annotations $remote_dir/results"
rsync -avP data/videos/ "$dest:$remote_dir/data/videos/"
rsync -avP data/gt/ "$dest:$remote_dir/data/gt/"
rsync -avP data/annotations/ "$dest:$remote_dir/data/annotations/"
if [ "${WITH_RESULTS:-0}" = 1 ]; then
  rsync -avP --exclude '_*' results/ "$dest:$remote_dir/results/"
fi
echo "done. data/frames/ is a cache and is rebuilt automatically on the destination."
