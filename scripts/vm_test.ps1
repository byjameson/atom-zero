param(
  [string]$VmUser = "byjameson",
  [string]$VmHost = "192.168.121.128",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
  [string]$RepoPath = "~/atom-zero",
  [switch]$Sync
)

$ErrorActionPreference = "Stop"
$sshArgs = @("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "-i", $KeyPath)

if ($Sync) {
  $local = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
  & scp @sshArgs -q -r "$local\src" "${VmUser}@${VmHost}:${RepoPath}/"
  & scp @sshArgs -q -r "$local\tests" "${VmUser}@${VmHost}:${RepoPath}/"
  & scp @sshArgs -q -r "$local\research" "${VmUser}@${VmHost}:${RepoPath}/"
  & scp @sshArgs -q "$local\pyproject.toml" "${VmUser}@${VmHost}:${RepoPath}/"
}

$script = @"
cd $RepoPath
PYTHONPATH=src .venv/bin/python -m pytest tests -q
PYTHONPATH=src .venv/bin/python benchmarks/self_improvement_regression.py
"@
$script = $script -replace "`r", ""
$script | & ssh @sshArgs "${VmUser}@${VmHost}" "tr -d '\r' | bash -s"
