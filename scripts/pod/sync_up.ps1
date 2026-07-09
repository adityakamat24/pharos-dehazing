# Sync the repo (code only) to a vast.ai pod.
# Usage: .\scripts\pod\sync_up.ps1 -SshHost ssh4.vast.ai -SshPort 12345
param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [Parameter(Mandatory = $true)][int]$SshPort,
    [string]$Dest = "/workspace/pharos"
)
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Write-Host "Syncing $repo -> root@${SshHost}:${Dest} (port $SshPort)"
# tar over ssh: robust on Windows without rsync; excludes heavy/local-only dirs
ssh -p $SshPort "root@$SshHost" "mkdir -p $Dest"
tar -C $repo --exclude=.venv --exclude=data --exclude=runs --exclude=.git -czf - . |
    ssh -p $SshPort "root@$SshHost" "tar -xzf - -C $Dest"
Write-Host "Done."
