param(
  [Parameter(Mandatory = $true)][string]$PlanPath,
  [Parameter(Mandatory = $true)][string]$StatusPath
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runner = Join-Path $repoRoot "paper\run_experiment_round.py"

$argList = @(
  $runner,
  "--plan", $PlanPath,
  "--status", $StatusPath
)

$proc = Start-Process -FilePath "python" -ArgumentList $argList -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
$launch = @{
  plan_path = (Resolve-Path $PlanPath).Path
  status_path = [System.IO.Path]::GetFullPath($StatusPath)
  pid = $proc.Id
  started_at = (Get-Date).ToUniversalTime().ToString("o")
}
$launch | ConvertTo-Json -Depth 4
