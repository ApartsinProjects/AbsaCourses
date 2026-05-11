$ErrorActionPreference = "Stop"

$repoRoot = "E:\Projects\CourseABSA"
$validationDir = Join-Path $repoRoot "paper\validation"
$realismOut = Join-Path $validationDir "realism_cycle_sequence_current.out.log"
$realismErr = Join-Path $validationDir "realism_cycle_sequence_current.err.log"
$pollOut = Join-Path $validationDir "realism_poll_current.out.log"
$pollErr = Join-Path $validationDir "realism_poll_current.err.log"

foreach ($path in @($realismOut, $realismErr, $pollOut, $pollErr)) {
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

Get-ChildItem $validationDir -Filter "prompt_debug_cycle_*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
foreach ($path in @(
    (Join-Path $validationDir "realism_poll_latest.json"),
    (Join-Path $validationDir "realism_poll_log.jsonl")
)) {
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

$realism = Start-Process -FilePath python `
    -ArgumentList "E:\Projects\CourseABSA\paper\realism_validation_experiment.py","run-cycle-sequence","--sample-size","32" `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $realismOut `
    -RedirectStandardError $realismErr `
    -PassThru

Start-Sleep -Seconds 2

$poller = Start-Process -FilePath python `
    -ArgumentList "E:\Projects\CourseABSA\paper\poll_realism_progress.py","--loop","--interval-seconds","300" `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $pollOut `
    -RedirectStandardError $pollErr `
    -PassThru

[pscustomobject]@{
    realism_pid = $realism.Id
    poller_pid = $poller.Id
    realism_stdout = $realismOut
    realism_stderr = $realismErr
    poller_stdout = $pollOut
    poller_stderr = $pollErr
    poll_snapshot = (Join-Path $validationDir "realism_poll_latest.json")
    poll_log = (Join-Path $validationDir "realism_poll_log.jsonl")
} | ConvertTo-Json -Depth 2
