<#
.SYNOPSIS
    Launch Pharos training on Windows using the project virtualenv.

.DESCRIPTION
    Activates D:\dehazing_desmoking\.venv, puts src/ on PYTHONPATH (the package
    is a src-layout, not necessarily pip-installed), and runs the engine trainer
    for the named experiment.

.PARAMETER Experiment
    Experiment name matching a file configs/<Experiment>.yaml (e.g. overfit50,
    local_sanity, full).

.PARAMETER Resume
    Optional checkpoint path, or "last" to resume the experiment's last.pt.

.PARAMETER Override
    Optional dotted key=value overrides passed straight to the trainer, e.g.
    -Override train.lr=1e-4,train.batch=4

.EXAMPLE
    .\scripts\run_train.ps1 -Experiment overfit50
    .\scripts\run_train.ps1 -Experiment local_sanity -Override train.batch=6
    .\scripts\run_train.ps1 -Experiment full -Resume last
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Experiment,
    [string]$Resume,
    [string[]]$Override
)

$ErrorActionPreference = "Stop"

$Root = "D:\dehazing_desmoking"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"

# Resolve the config relative to the repo/worktree this script lives in.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Config = Join-Path $RepoRoot "configs\$Experiment.yaml"

if (-not (Test-Path $Python)) { throw "venv python not found at $Python" }
if (-not (Test-Path $Config)) { throw "config not found: $Config" }

if (Test-Path $Activate) { . $Activate }

# src-layout: make `pharos` importable without an editable install.
$env:PYTHONPATH = (Join-Path $RepoRoot "src") + [IO.Path]::PathSeparator + $env:PYTHONPATH

$cmd = @("-m", "pharos.engine.train", "--config", $Config)
if ($Resume) { $cmd += @("--resume", $Resume) }
if ($Override) { $cmd += @("--override") + $Override }

Write-Host "Launching: $Python $($cmd -join ' ')" -ForegroundColor Cyan
& $Python @cmd
exit $LASTEXITCODE
