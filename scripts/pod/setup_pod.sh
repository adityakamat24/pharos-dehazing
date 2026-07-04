#!/usr/bin/env bash
# One-time pod setup. Run from anywhere: bash /workspace/pharos/scripts/pod/setup_pod.sh
set -euo pipefail

cd /workspace/pharos
apt-get update -qq && apt-get install -y -qq rsync tmux libgl1 libglib2.0-0 > /dev/null
python -m pip install -q -e .[train,export]
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
mkdir -p /workspace/data /workspace/runs
echo "Pod ready. Next: download datasets, then start training under tmux."
