param(
  [int]$OfferId = 0,
  [double]$Dph = 0.2,
  [int]$Layers = 24,
  [string]$Seeds = "101 102",
  [int]$Steps = 4000,
  [int]$Batch = 16,
  [double]$MaxHours = 3.0,
  [switch]$Execute
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

Write-Host "== preflight =="
python scripts/vastai/preflight.py --max-price 0.25 --hours $MaxHours
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== bundle =="
python scripts/vastai/bundle.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$launchArgs = @(
  "scripts/vastai/launch.py",
  "--offer-id", "$OfferId",
  "--dph", "$Dph",
  "--layers", "$Layers",
  "--seeds", "$Seeds",
  "--steps", "$Steps",
  "--batch", "$Batch",
  "--max-hours", "$MaxHours"
)
if ($Execute) { $launchArgs += "--execute" }

Write-Host "== launch =="
python @launchArgs
exit $LASTEXITCODE
