$ErrorActionPreference = "Stop"

$repo = "E:\Projects\CourseABSA"
$logDir = Join-Path $repo "paper\batch_requests\monitor_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMddTHHmmss"
$logPath = Join-Path $logDir "monitor_$timestamp.log"

Set-Location $repo
python "E:\Projects\CourseABSA\paper\monitor_generation_job.py" *>&1 | Tee-Object -FilePath $logPath
