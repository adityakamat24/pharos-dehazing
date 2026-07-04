# vast.ai pod workflow (billed — start instances only with explicit user approval)

Principle: everything is prepared and sanity-tested locally on the RTX 3060 first; the pod only runs
the long training job. Keep pod time minimal.

## 1. Pick an instance
24GB VRAM is plenty (batch 32 @ 256px). Good value targets: RTX 3090/4090 or A5000.
```
vastai search offers "gpu_name in [RTX_3090,RTX_4090,A5000] num_gpus=1 inet_down>200" -o dph
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 120 --ssh
vastai show instances    # get ssh host/port
```

## 2. Sync code (not data) and set up
```
scripts/pod/sync_up.ps1 -SshHost <host> -SshPort <port>     # rsync/scp repo w/o data, runs, .venv
ssh -p <port> root@<host> "bash /workspace/pharos/scripts/pod/setup_pod.sh"
```

## 3. Pull datasets directly on the pod (fast pod internet; never upload from home)
```
ssh -p <port> root@<host> "python /workspace/pharos/scripts/download_datasets.py --all --data-root /workspace/data"
```
Pod-only large sets (RESIDE-OTS, HazeWorld, RS-Haze) are listed in DESIGN.md §6.

## 4. Train under tmux, monitor, pull checkpoints
```
ssh ... "tmux new -d -s train 'cd /workspace/pharos && python -m pharos.engine.train --config configs/full.yaml --override data_root=/workspace/data out_root=/workspace/runs'"
scripts/pod/pull_ckpt.ps1 -SshHost <host> -SshPort <port>   # rsync runs/<exp>/ckpt + eval back
```

## 5. DESTROY the instance when done (stopping still bills storage)
```
vastai destroy instance <ID>
```
