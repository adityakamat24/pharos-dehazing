# Pull checkpoints + eval results back from a vast.ai pod.
# Usage: .\scripts\pod\pull_ckpt.ps1 -SshHost ssh4.vast.ai -SshPort 12345 -Exp full
param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [Parameter(Mandatory = $true)][int]$SshPort,
    [string]$Exp = "full",
    [string]$Remote = "/workspace/runs"
)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$dest = Join-Path $repo "runs\$Exp"
New-Item -ItemType Directory -Force $dest | Out-Null
# Latest checkpoint + eval artifacts only (not optimizer-heavy intermediates)
scp -P $SshPort "root@${SshHost}:$Remote/$Exp/ckpt/latest*.pth" $dest 2>$null
scp -r -P $SshPort "root@${SshHost}:$Remote/$Exp/eval" $dest 2>$null
Write-Host "Pulled into $dest"
